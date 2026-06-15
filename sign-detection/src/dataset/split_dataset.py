"""Dataset splitting utilities for factory sign detection."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


SPLITS = ("train", "val", "test")
CLASS_COLUMNS = (
    "M014_Helmet",
    "M015_Vest",
    "P004_NoThoroughfare",
    "W011_Slippery",
)


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if any(ratio < 0 for ratio in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("Split ratios must be non-negative.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")


def _split_dirs(output_splits_dir: Path) -> list[Path]:
    return [
        output_splits_dir / split / kind
        for split in SPLITS
        for kind in ("images", "labels")
    ]


def _has_existing_split_files(output_splits_dir: Path) -> bool:
    for directory in _split_dirs(output_splits_dir):
        if directory.exists() and any(path.is_file() for path in directory.iterdir()):
            return True
    return False


def _prepare_output_dirs(output_splits_dir: Path, overwrite: bool) -> None:
    if _has_existing_split_files(output_splits_dir):
        if not overwrite:
            raise FileExistsError(
                f"Split output folders under {output_splits_dir} already contain files. "
                "Set overwrite=True to regenerate them."
            )
        for directory in _split_dirs(output_splits_dir):
            if directory.exists():
                shutil.rmtree(directory)

    for directory in _split_dirs(output_splits_dir):
        directory.mkdir(parents=True, exist_ok=True)


def _counts_for_ratios(n_items: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, int]:
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    raw = {split: n_items * ratio for split, ratio in ratios.items()}
    counts = {split: int(value) for split, value in raw.items()}
    remainder = n_items - sum(counts.values())
    order = sorted(SPLITS, key=lambda split: (raw[split] - counts[split], ratios[split]), reverse=True)
    for split in order[:remainder]:
        counts[split] += 1
    return counts


def _assign_random(profile_df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> pd.Series:
    rng = random.Random(seed)
    indices = list(profile_df.index)
    rng.shuffle(indices)
    counts = _counts_for_ratios(len(indices), train_ratio, val_ratio, test_ratio)
    assignments: dict[int, str] = {}
    cursor = 0
    for split in SPLITS:
        for index in indices[cursor: cursor + counts[split]]:
            assignments[index] = split
        cursor += counts[split]
    return pd.Series(assignments)


def _assign_stratified(
    profile_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.Series:
    rng = random.Random(seed)
    assignments: dict[int, str] = {}
    for group_key, group_df in profile_df.groupby("split_group_key", sort=True):
        indices = list(group_df.index)
        rng.shuffle(indices)
        counts = _counts_for_ratios(len(indices), train_ratio, val_ratio, test_ratio)
        cursor = 0
        for split in SPLITS:
            for index in indices[cursor: cursor + counts[split]]:
                assignments[index] = split
            cursor += counts[split]
    return pd.Series(assignments)


def _should_use_stratification(profile_df: pd.DataFrame) -> tuple[bool, str]:
    if "split_group_key" not in profile_df.columns:
        return False, "split_group_key missing"
    group_sizes = profile_df["split_group_key"].value_counts()
    if group_sizes.empty:
        return False, "no split groups available"
    tiny_groups = int((group_sizes < len(SPLITS)).sum())
    if tiny_groups > 0:
        return False, f"{tiny_groups} split_group_key groups have fewer than {len(SPLITS)} samples"
    return True, "stratified by split_group_key"


def _copy_pair(row: pd.Series, input_images_dir: Path, input_labels_dir: Path, output_splits_dir: Path) -> dict[str, Any]:
    image_name = str(row["image_name"])
    base_name = str(row.get("base_name", Path(image_name).stem))
    label_name = str(row.get("label_name", "")) if "label_name" in row else ""
    if not label_name or label_name == "nan":
        label_name = f"{base_name}.txt"

    source_image = input_images_dir / image_name
    source_label = input_labels_dir / label_name
    split = str(row["split"])
    target_image = output_splits_dir / split / "images" / image_name
    target_label = output_splits_dir / split / "labels" / label_name

    status = "copied"
    notes: list[str] = []
    if not source_image.exists():
        status = "failed"
        notes.append("source image missing")
    if not source_label.exists():
        status = "failed"
        notes.append("source label missing")

    if status == "copied":
        shutil.copy2(source_image, target_image)
        shutil.copy2(source_label, target_label)

    return {
        "label_name": label_name,
        "source_image_path": str(source_image),
        "source_label_path": str(source_label),
        "target_image_path": str(target_image),
        "target_label_path": str(target_label),
        "status": status,
        "notes": "; ".join(notes),
    }


def split_dataset(
    image_profile_df: pd.DataFrame,
    input_images_dir: Path,
    input_labels_dir: Path,
    output_splits_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Create deterministic train/val/test YOLO splits from valid image profiles."""
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    input_images_dir = Path(input_images_dir)
    input_labels_dir = Path(input_labels_dir)
    output_splits_dir = Path(output_splits_dir)

    profile_df = image_profile_df.copy()
    if "status" in profile_df.columns:
        profile_df = profile_df[profile_df["status"] == "valid"].copy()
    if profile_df.empty:
        raise ValueError("No valid images are available for splitting.")

    _prepare_output_dirs(output_splits_dir, overwrite=overwrite)

    use_stratified, split_note = _should_use_stratification(profile_df)
    if use_stratified:
        assignments = _assign_stratified(profile_df, train_ratio, val_ratio, test_ratio, seed)
    else:
        assignments = _assign_random(profile_df, train_ratio, val_ratio, test_ratio, seed)

    profile_df["split"] = profile_df.index.map(assignments)
    profile_df["notes"] = split_note if use_stratified else f"stratification fallback used: {split_note}"

    records: list[dict[str, Any]] = []
    for _, row in profile_df.sort_values(["split", "image_name"]).iterrows():
        copy_info = _copy_pair(row, input_images_dir, input_labels_dir, output_splits_dir)
        row_record = row.to_dict()
        row_record.update(copy_info)
        records.append(row_record)

    required_columns = [
        "image_name",
        "label_name",
        "base_name",
        "split",
        "split_group_key",
        "is_no_sign",
        "has_M014_Helmet",
        "has_M015_Vest",
        "has_P004_NoThoroughfare",
        "has_W011_Slippery",
        "num_objects",
        "num_M014_Helmet",
        "num_M015_Vest",
        "num_P004_NoThoroughfare",
        "num_W011_Slippery",
        "source_image_path",
        "source_label_path",
        "target_image_path",
        "target_label_path",
        "status",
        "notes",
    ]
    split_df = pd.DataFrame(records)
    for column in required_columns:
        if column not in split_df.columns:
            split_df[column] = ""
    return split_df[required_columns]
