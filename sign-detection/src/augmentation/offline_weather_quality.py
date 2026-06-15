"""Weather and quality degradation augmentations for factory sign detection."""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from .offline_common import (
    copy_label_unchanged,
    empty_report,
    find_image_label_pairs,
    parse_yolo_labels,
    prepare_output_dirs,
    report_row,
    select_pairs,
)


def _jpeg_compress(image: Image.Image, rng: random.Random) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(35, 70))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _sensor_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    sigma = rng.uniform(4.0, 12.0)
    noise_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    noisy = np.clip(array + noise_rng.normal(0, sigma, array.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def _dirty_lens(image: Image.Image, rng: random.Random) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    for _ in range(rng.randint(3, 7)):
        radius = rng.randint(max(3, min(width, height) // 40), max(6, min(width, height) // 12))
        x = rng.randint(0, width)
        y = rng.randint(0, height)
        alpha = rng.randint(18, 42)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(230, 230, 225, alpha))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=5))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _low_resolution(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    factor = rng.uniform(0.45, 0.75)
    small_size = (max(16, int(width * factor)), max(16, int(height * factor)))
    resampling = getattr(Image, "Resampling", Image)
    small = image.resize(small_size, resampling.BILINEAR)
    return small.resize((width, height), resampling.BILINEAR)


def _apply_quality_transform(image: Image.Image, image_name: str, config: dict, seed: int) -> tuple[Image.Image, str]:
    rng = random.Random(f"{seed}:{image_name}:weather_quality")
    transforms = []
    if config.get("enable_blur", True):
        transforms.append(("gaussian_blur", lambda img: img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 1.8)))))
    if config.get("enable_compression", True):
        transforms.append(("jpeg_compression", lambda img: _jpeg_compress(img, rng)))
    if config.get("enable_noise", True):
        transforms.append(("sensor_noise", lambda img: _sensor_noise(img, rng)))
    if config.get("enable_rain_dirty_lens", True):
        transforms.append(("dirty_lens", lambda img: _dirty_lens(img, rng)))
    transforms.append(("low_resolution_upsample", lambda img: _low_resolution(img, rng)))

    rng.shuffle(transforms)
    selected = transforms[: rng.randint(1, min(3, len(transforms)))]
    augmented = image
    names = []
    for name, transform in selected:
        augmented = transform(augmented)
        names.append(name)
    return augmented, "+".join(names)


def generate_weather_quality_augmentation(
    images_dir: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    ratio: float,
    config: dict,
    seed: int = 42,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Generate train-only CCTV quality augmentations and copy labels unchanged."""
    pairs = select_pairs(find_image_label_pairs(images_dir, labels_dir), ratio, seed)
    if not pairs:
        return empty_report()

    prepare_output_dirs(output_images_dir, output_labels_dir)
    rows: list[dict[str, object]] = []
    for image_path, label_path in pairs:
        output_image = Path(output_images_dir) / f"quality_{image_path.stem}.jpg"
        output_label = Path(output_labels_dir) / f"quality_{image_path.stem}.txt"
        labels = parse_yolo_labels(label_path)

        if (output_image.exists() or output_label.exists()) and not overwrite:
            rows.append(report_row("weather_quality", image_path, label_path, output_image, output_label, "skipped", "output exists", len(labels), 0))
            continue

        try:
            with Image.open(image_path) as image:
                augmented, transform_name = _apply_quality_transform(image.convert("RGB"), image_path.name, config, seed)
                augmented.save(output_image, quality=90)
            # Quality transforms keep geometry fixed, so boxes remain valid unchanged.
            copy_label_unchanged(label_path, output_label)
            rows.append(
                report_row(
                    "weather_quality",
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
            rows.append(report_row("weather_quality", image_path, label_path, output_image, output_label, "failed", str(exc), len(labels), 0))

    return pd.DataFrame(rows)
