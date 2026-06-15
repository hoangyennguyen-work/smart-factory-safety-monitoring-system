"""YOLO dataset validation utilities for factory sign detection."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError


CLASS_COLUMNS: dict[int, str] = {
    0: "M014_Helmet",
    1: "M015_Vest",
    2: "P004_NoThoroughfare",
    3: "W011_Slippery",
}
IGNORED_PLACEHOLDER_FILENAMES = {".gitkeep"}


def _empty_counts() -> dict[str, int | bool]:
    counts: dict[str, int | bool] = {}
    for class_name in CLASS_COLUMNS.values():
        counts[f"num_{class_name}"] = 0
        counts[f"has_{class_name}"] = False
    return counts


def _base_record(
    image_path: Path | None,
    label_path: Path | None,
    base_name: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "image_name": image_path.name if image_path else "",
        "label_name": label_path.name if label_path else "",
        "base_name": base_name,
        "image_path": str(image_path) if image_path else "",
        "label_path": str(label_path) if label_path else "",
        "status": "valid",
        "errors": "",
        "warnings": "",
        "image_width": None,
        "image_height": None,
        "num_objects": 0,
        "is_no_sign": True,
    }
    record.update(_empty_counts())
    return record


def _read_image_size(image_path: Path) -> tuple[int | None, int | None, str | None]:
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
        return int(width), int(height), None
    except (OSError, UnidentifiedImageError) as exc:
        return None, None, f"unreadable_image: {exc}"


def _parse_label_file(
    label_path: Path,
    width: int,
    height: int,
    class_ids: set[int],
    boundary_tolerance: float,
) -> tuple[Counter[int], list[str]]:
    counts: Counter[int] = Counter()
    errors: list[str] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return counts, errors

    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: invalid_yolo_row expected 5 values")
            continue

        class_text, *coord_texts = parts
        try:
            class_float = float(class_text)
            class_id = int(class_float)
        except ValueError:
            errors.append(f"line {line_number}: class_id is not numeric")
            continue

        if class_float != class_id:
            errors.append(f"line {line_number}: class_id must be an integer")
            continue
        if class_id not in class_ids:
            errors.append(f"line {line_number}: invalid_class_id {class_id}")
            continue

        try:
            x_center, y_center, box_width, box_height = [float(value) for value in coord_texts]
        except ValueError:
            errors.append(f"line {line_number}: yolo coordinates are not numeric")
            continue

        values = {
            "x_center": x_center,
            "y_center": y_center,
            "width": box_width,
            "height": box_height,
        }
        out_of_range = [name for name, value in values.items() if value < 0.0 or value > 1.0]
        if out_of_range:
            errors.append(f"line {line_number}: coordinates outside [0,1]: {', '.join(out_of_range)}")
            continue
        if box_width <= 0.0 or box_height <= 0.0:
            errors.append(f"line {line_number}: width and height must be greater than 0")
            continue

        left = (x_center - box_width / 2.0) * width
        right = (x_center + box_width / 2.0) * width
        top = (y_center - box_height / 2.0) * height
        bottom = (y_center + box_height / 2.0) * height
        x_tol = boundary_tolerance * width
        y_tol = boundary_tolerance * height
        if left < -x_tol or top < -y_tol or right > width + x_tol or bottom > height + y_tol:
            errors.append(f"line {line_number}: invalid_bbox outside image boundaries")
            continue

        counts[class_id] += 1

    return counts, errors


def _image_candidates(images_dir: Path, extensions: set[str]) -> tuple[dict[str, Path], dict[str, list[Path]], list[Path]]:
    supported: dict[str, list[Path]] = defaultdict(list)
    unsupported: list[Path] = []
    for path in sorted(images_dir.iterdir()) if images_dir.exists() else []:
        if not path.is_file():
            continue
        if path.name in IGNORED_PLACEHOLDER_FILENAMES:
            continue
        if path.suffix.lower() in extensions:
            supported[path.stem].append(path)
        else:
            unsupported.append(path)
    unique = {stem: paths[0] for stem, paths in supported.items() if len(paths) == 1}
    duplicates = {stem: paths for stem, paths in supported.items() if len(paths) > 1}
    return unique, duplicates, unsupported


def validate_yolo_dataset(
    images_dir: Path,
    labels_dir: Path,
    class_ids: set[int],
    image_extensions: list[str],
    boundary_tolerance: float = 1e-4,
) -> pd.DataFrame:
    """Validate a prepared YOLO input dataset without modifying files.

    Empty label files are valid no-sign images. The returned DataFrame has one
    row per discovered image and one row per unmatched label.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    extensions = {extension.lower() for extension in image_extensions}
    records: list[dict[str, Any]] = []

    image_by_stem, duplicate_images, unsupported_images = _image_candidates(images_dir, extensions)
    label_by_stem = {
        path.stem: path
        for path in sorted(labels_dir.glob("*.txt")) if labels_dir.exists() and path.is_file()
    }

    for image_path in unsupported_images:
        record = _base_record(image_path=image_path, label_path=None, base_name=image_path.stem)
        record["status"] = "invalid"
        record["errors"] = f"unsupported_image_extension: {image_path.suffix}"
        records.append(record)

    for stem, paths in duplicate_images.items():
        label_path = label_by_stem.get(stem)
        for image_path in paths:
            record = _base_record(image_path=image_path, label_path=label_path, base_name=stem)
            record["status"] = "invalid"
            record["errors"] = "duplicate_base_name"
            record["warnings"] = "multiple supported image files share this base name"
            records.append(record)

    all_stems = sorted(set(image_by_stem) | set(label_by_stem))
    duplicate_stems = set(duplicate_images)
    for stem in all_stems:
        if stem in duplicate_stems:
            continue

        image_path = image_by_stem.get(stem)
        label_path = label_by_stem.get(stem)
        record = _base_record(image_path=image_path, label_path=label_path, base_name=stem)
        errors: list[str] = []

        if image_path is None:
            errors.append("missing_image")
        if label_path is None:
            errors.append("missing_label")

        if image_path is not None:
            width, height, image_error = _read_image_size(image_path)
            record["image_width"] = width
            record["image_height"] = height
            if image_error:
                errors.append(image_error)

        if label_path is not None and record["image_width"] and record["image_height"]:
            try:
                counts, label_errors = _parse_label_file(
                    label_path=label_path,
                    width=int(record["image_width"]),
                    height=int(record["image_height"]),
                    class_ids=class_ids,
                    boundary_tolerance=boundary_tolerance,
                )
            except OSError as exc:
                counts = Counter()
                label_errors = [f"label_read_error: {exc}"]

            errors.extend(label_errors)
            record["num_objects"] = int(sum(counts.values()))
            record["is_no_sign"] = record["num_objects"] == 0
            for class_id, class_name in CLASS_COLUMNS.items():
                count = int(counts[class_id])
                record[f"num_{class_name}"] = count
                record[f"has_{class_name}"] = count > 0

        if errors:
            record["status"] = "invalid"
            record["errors"] = " | ".join(errors)

        records.append(record)

    columns = [
        "image_name",
        "label_name",
        "base_name",
        "image_path",
        "label_path",
        "status",
        "errors",
        "warnings",
        "image_width",
        "image_height",
        "num_objects",
        "num_M014_Helmet",
        "num_M015_Vest",
        "num_P004_NoThoroughfare",
        "num_W011_Slippery",
        "is_no_sign",
        "has_M014_Helmet",
        "has_M015_Vest",
        "has_P004_NoThoroughfare",
        "has_W011_Slippery",
    ]
    return pd.DataFrame(records, columns=columns)
