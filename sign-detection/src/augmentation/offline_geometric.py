"""Geometry-changing offline augmentations that must update bounding boxes."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .offline_common import (
    empty_report,
    find_image_label_pairs,
    parse_yolo_labels,
    prepare_output_dirs,
    report_row,
    select_pairs,
    write_yolo_labels,
)


def _affine_matrix(width: int, height: int, angle_deg: float, scale: float, tx: float, ty: float) -> np.ndarray:
    """Build a forward affine matrix from original pixels to augmented pixels."""
    angle = math.radians(angle_deg)
    cos_a = scale * math.cos(angle)
    sin_a = scale * math.sin(angle)
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [
            [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy + tx],
            [sin_a, cos_a, -sin_a * cx + (1 - cos_a) * cy + ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _transform_image(image: Image.Image, matrix: np.ndarray) -> Image.Image:
    """Apply an affine transform to an image while keeping the original canvas size."""
    inverse = np.linalg.inv(matrix)
    coeffs = tuple(inverse[:2, :].reshape(-1))
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    return image.transform(image.size, Image.Transform.AFFINE, coeffs, resample=resampling, fillcolor=(114, 114, 114))


def _bbox_to_corners(label: tuple[int, float, float, float, float], width: int, height: int) -> np.ndarray:
    """Convert one normalized YOLO box to four pixel-space corner points."""
    _, x_center, y_center, box_w, box_h = label
    x = x_center * width
    y = y_center * height
    half_w = box_w * width / 2.0
    half_h = box_h * height / 2.0
    return np.array(
        [
            [x - half_w, y - half_h, 1.0],
            [x + half_w, y - half_h, 1.0],
            [x + half_w, y + half_h, 1.0],
            [x - half_w, y + half_h, 1.0],
        ],
        dtype=np.float64,
    )


def _transform_labels(
    labels: list[tuple[int, float, float, float, float]],
    matrix: np.ndarray,
    width: int,
    height: int,
) -> list[tuple[int, float, float, float, float]]:
    """Transform YOLO boxes through the affine matrix and clip them to the image."""
    transformed: list[tuple[int, float, float, float, float]] = []
    for label in labels:
        class_id = label[0]
        corners = _bbox_to_corners(label, width, height)
        new_corners = (matrix @ corners.T).T[:, :2]

        # Clip transformed corners to the image canvas before making the enclosing box.
        x_min = float(np.clip(new_corners[:, 0].min(), 0, width))
        x_max = float(np.clip(new_corners[:, 0].max(), 0, width))
        y_min = float(np.clip(new_corners[:, 1].min(), 0, height))
        y_max = float(np.clip(new_corners[:, 1].max(), 0, height))
        box_w = x_max - x_min
        box_h = y_max - y_min
        area_norm = (box_w * box_h) / (width * height)

        # Drop boxes that collapsed after transform/clipping.
        if box_w < 3 or box_h < 3 or area_norm < 1e-6:
            continue

        x_center = (x_min + x_max) / 2.0 / width
        y_center = (y_min + y_max) / 2.0 / height
        transformed.append((class_id, x_center, y_center, box_w / width, box_h / height))
    return transformed


def _params_for_image(image_path: Path, config: dict, seed: int) -> tuple[float, float, float, float]:
    rng = random.Random(f"{seed}:{image_path.name}:geometric")
    max_rotation = float(config.get("max_rotation_degrees", 25))
    min_scale = float(config.get("min_scale", 0.70))
    max_scale = float(config.get("max_scale", 1.30))
    max_translate = float(config.get("max_translate", 0.12))
    angle = rng.uniform(-max_rotation, max_rotation)
    scale = rng.uniform(min_scale, max_scale)
    tx_fraction = rng.uniform(-max_translate, max_translate)
    ty_fraction = rng.uniform(-max_translate, max_translate)
    return angle, scale, tx_fraction, ty_fraction


def generate_geometric_augmentation(
    images_dir: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    ratio: float,
    config: dict,
    seed: int = 42,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Generate train-only geometric sign augmentations and update YOLO boxes."""
    pairs = select_pairs(find_image_label_pairs(images_dir, labels_dir), ratio, seed)
    if not pairs:
        return empty_report()

    prepare_output_dirs(output_images_dir, output_labels_dir)
    rows: list[dict[str, object]] = []
    for image_path, label_path in pairs:
        output_image = Path(output_images_dir) / f"geo_{image_path.stem}.jpg"
        output_label = Path(output_labels_dir) / f"geo_{image_path.stem}.txt"
        labels = parse_yolo_labels(label_path)

        if (output_image.exists() or output_label.exists()) and not overwrite:
            rows.append(
                report_row("geometric", image_path, label_path, output_image, output_label, "skipped", "output exists", len(labels), 0)
            )
            continue

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                angle, scale, tx_fraction, ty_fraction = _params_for_image(image_path, config, seed)
                matrix = _affine_matrix(width, height, angle, scale, tx_fraction * width, ty_fraction * height)
                augmented = _transform_image(image, matrix)
                augmented_labels = _transform_labels(labels, matrix, width, height)

                if labels and not augmented_labels:
                    rows.append(
                        report_row(
                            "geometric",
                            image_path,
                            label_path,
                            output_image,
                            output_label,
                            "skipped",
                            "all boxes became invalid or too tiny after geometric transform",
                            len(labels),
                            0,
                        )
                    )
                    continue

                augmented.save(output_image, quality=95)
                # Geometry changes object positions, so labels must be transformed instead of copied.
                write_yolo_labels(output_label, augmented_labels)
                rows.append(
                    report_row(
                        "geometric",
                        image_path,
                        label_path,
                        output_image,
                        output_label,
                        "generated",
                        "affine rotation/scale/translation; perspective skipped for reliability",
                        len(labels),
                        len(augmented_labels),
                    )
                )
        except Exception as exc:
            rows.append(
                report_row("geometric", image_path, label_path, output_image, output_label, "failed", str(exc), len(labels), 0)
            )

    return pd.DataFrame(rows)
