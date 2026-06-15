"""Shared helpers for offline sign-detection augmentation."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_image_label_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Return sorted image-label pairs from a YOLO image/label folder pair."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.iterdir()) if images_dir.exists() else []:
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def select_pairs(pairs: list[tuple[Path, Path]], ratio: float, seed: int) -> list[tuple[Path, Path]]:
    """Select a deterministic subset of pairs from an augmentation ratio."""
    if ratio <= 0 or not pairs:
        return []
    rng = random.Random(seed)
    count = int(round(len(pairs) * ratio))
    count = max(1, min(len(pairs), count))
    selected = pairs.copy()
    rng.shuffle(selected)
    return sorted(selected[:count], key=lambda pair: pair[0].name)


def parse_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse YOLO labels as class_id, x_center, y_center, width, height tuples."""
    text = Path(label_path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    labels = []
    for line in text.splitlines():
        class_text, x_text, y_text, w_text, h_text = line.split()
        labels.append((int(float(class_text)), float(x_text), float(y_text), float(w_text), float(h_text)))
    return labels


def write_yolo_labels(label_path: Path, labels: Iterable[tuple[int, float, float, float, float]]) -> None:
    """Write YOLO labels using stable six-decimal formatting."""
    lines = [
        f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        for class_id, x, y, w, h in labels
    ]
    Path(label_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prepare_output_dirs(output_images_dir: Path, output_labels_dir: Path) -> None:
    """Create augmentation output folders."""
    Path(output_images_dir).mkdir(parents=True, exist_ok=True)
    Path(output_labels_dir).mkdir(parents=True, exist_ok=True)


def copy_label_unchanged(source_label: Path, target_label: Path) -> None:
    """Copy a YOLO label file unchanged for non-geometric image transforms."""
    shutil.copy2(source_label, target_label)


def report_row(
    augmentation_type: str,
    image_path: Path | str,
    label_path: Path | str,
    output_image_path: Path | str,
    output_label_path: Path | str,
    status: str,
    notes: str,
    num_original_objects: int,
    num_augmented_objects: int,
) -> dict[str, object]:
    """Create one standardized augmentation report row."""
    return {
        "augmentation_type": augmentation_type,
        "original_image_path": str(image_path),
        "original_label_path": str(label_path),
        "augmented_image_path": str(output_image_path),
        "augmented_label_path": str(output_label_path),
        "status": status,
        "notes": notes,
        "num_original_objects": int(num_original_objects),
        "num_augmented_objects": int(num_augmented_objects),
    }


def empty_report() -> pd.DataFrame:
    """Return an empty report with the standard augmentation schema."""
    return pd.DataFrame(
        columns=[
            "augmentation_type",
            "original_image_path",
            "original_label_path",
            "augmented_image_path",
            "augmented_label_path",
            "status",
            "notes",
            "num_original_objects",
            "num_augmented_objects",
        ]
    )
