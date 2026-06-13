"""Photometric offline augmentations for factory sign detection."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .offline_common import (
    copy_label_unchanged,
    empty_report,
    find_image_label_pairs,
    parse_yolo_labels,
    prepare_output_dirs,
    report_row,
    select_pairs,
)


def _ir_style(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    rgb = Image.merge("RGB", (gray, gray, gray))
    return ImageEnhance.Contrast(rgb).enhance(1.20)


def _sunlight(image: Image.Image, rng: random.Random) -> Image.Image:
    bright = ImageEnhance.Brightness(image).enhance(rng.uniform(1.15, 1.35))
    bright = ImageEnhance.Contrast(bright).enhance(rng.uniform(1.05, 1.20))
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    radius = int(min(width, height) * rng.uniform(0.18, 0.35))
    cx = rng.randint(0, width)
    cy = rng.randint(0, max(1, height // 2))
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 245, 210, 55))
    return Image.alpha_composite(bright.convert("RGBA"), overlay).convert("RGB")


def _low_light(image: Image.Image, rng: random.Random) -> Image.Image:
    dark = ImageEnhance.Brightness(image).enhance(rng.uniform(0.45, 0.70))
    return ImageEnhance.Contrast(dark).enhance(rng.uniform(0.85, 1.10))


def _shadow(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    x0 = rng.randint(-width // 2, width // 2)
    x1 = x0 + rng.randint(width // 3, width)
    draw.polygon([(x0, 0), (x1, 0), (x1 + width // 4, height), (x0 + width // 4, height)], fill=95)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(8, min(width, height) // 12)))
    shadow = Image.new("RGB", image.size, (0, 0, 0))
    return Image.composite(shadow, image, mask).convert("RGB")


def _apply_photometric_transform(image: Image.Image, image_name: str, config: dict, seed: int) -> tuple[Image.Image, str]:
    rng = random.Random(f"{seed}:{image_name}:photometric")
    enabled = []
    if config.get("enable_ir", True):
        enabled.append(("ir_grayscale", _ir_style))
    if config.get("enable_sunlight", True):
        enabled.append(("sunlight", lambda img: _sunlight(img, rng)))
    if config.get("enable_low_light", True):
        enabled.append(("low_light", lambda img: _low_light(img, rng)))
    if config.get("enable_shadow", True):
        enabled.append(("shadow", lambda img: _shadow(img, rng)))

    if not enabled:
        return image, "no photometric transforms enabled"
    name, transform = rng.choice(enabled)
    adjusted = transform(image)
    adjusted = ImageEnhance.Color(adjusted).enhance(rng.uniform(0.85, 1.10))
    return adjusted, name


def generate_photometric_augmentation(
    images_dir: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    ratio: float,
    config: dict,
    seed: int = 42,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Generate train-only photometric augmentations and copy labels unchanged."""
    pairs = select_pairs(find_image_label_pairs(images_dir, labels_dir), ratio, seed)
    if not pairs:
        return empty_report()

    prepare_output_dirs(output_images_dir, output_labels_dir)
    rows: list[dict[str, object]] = []
    for image_path, label_path in pairs:
        output_image = Path(output_images_dir) / f"photo_{image_path.stem}.jpg"
        output_label = Path(output_labels_dir) / f"photo_{image_path.stem}.txt"
        labels = parse_yolo_labels(label_path)

        if (output_image.exists() or output_label.exists()) and not overwrite:
            rows.append(report_row("photometric", image_path, label_path, output_image, output_label, "skipped", "output exists", len(labels), 0))
            continue

        try:
            with Image.open(image_path) as image:
                augmented, transform_name = _apply_photometric_transform(image.convert("RGB"), image_path.name, config, seed)
                augmented.save(output_image, quality=95)
            # Photometric transforms do not move objects, so YOLO labels remain valid byte-for-byte.
            copy_label_unchanged(label_path, output_label)
            rows.append(
                report_row(
                    "photometric",
                    image_path,
                    label_path,
                    output_image,
                    output_label,
                    "generated",
                    f"label copied unchanged; transform={transform_name}",
                    len(labels),
                    len(labels),
                )
            )
        except Exception as exc:
            rows.append(report_row("photometric", image_path, label_path, output_image, output_label, "failed", str(exc), len(labels), 0))

    return pd.DataFrame(rows)
