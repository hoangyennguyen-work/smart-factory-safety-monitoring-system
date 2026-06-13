"""Minimal pre-split profiling utilities for factory sign detection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


CLASS_COLUMNS: dict[int, str] = {
    0: "M014_Helmet",
    1: "M015_Vest",
    2: "P004_NoThoroughfare",
    3: "W011_Slippery",
}


def _split_group_key(row: pd.Series) -> str:
    return (
        f"M014={int(bool(row['has_M014_Helmet']))}|"
        f"M015={int(bool(row['has_M015_Vest']))}|"
        f"P004={int(bool(row['has_P004_NoThoroughfare']))}|"
        f"W011={int(bool(row['has_W011_Slippery']))}|"
        f"NONE={int(bool(row['is_no_sign']))}"
    )


def build_image_level_profile(validation_df: pd.DataFrame) -> pd.DataFrame:
    """Create one valid-image row for Notebook 02 stratified splitting."""
    valid = validation_df[validation_df["status"] == "valid"].copy()
    if valid.empty:
        columns = [
            "image_name",
            "base_name",
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
            "image_width",
            "image_height",
            "aspect_ratio",
            "split_group_key",
        ]
        return pd.DataFrame(columns=columns)

    valid["aspect_ratio"] = valid["image_width"].astype(float) / valid["image_height"].astype(float)
    valid["split_group_key"] = valid.apply(_split_group_key, axis=1)
    columns = [
        "image_name",
        "base_name",
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
        "image_width",
        "image_height",
        "aspect_ratio",
        "split_group_key",
    ]
    return valid[columns].reset_index(drop=True)


def _parse_valid_label_rows(row: pd.Series) -> list[dict[str, Any]]:
    label_path = Path(str(row["label_path"]))
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    records: list[dict[str, Any]] = []
    image_width = int(row["image_width"])
    image_height = int(row["image_height"])
    for line in text.splitlines():
        class_text, x_text, y_text, width_text, height_text = line.split()
        class_id = int(float(class_text))
        x_center = float(x_text)
        y_center = float(y_text)
        width_norm = float(width_text)
        height_norm = float(height_text)
        box_width_px = width_norm * image_width
        box_height_px = height_norm * image_height
        records.append(
            {
                "image_name": row["image_name"],
                "class_id": class_id,
                "class_name": CLASS_COLUMNS.get(class_id, str(class_id)),
                "x_center_norm": x_center,
                "y_center_norm": y_center,
                "width_norm": width_norm,
                "height_norm": height_norm,
                "box_area_norm": width_norm * height_norm,
                "image_width": image_width,
                "image_height": image_height,
                "box_width_px": box_width_px,
                "box_height_px": box_height_px,
                "box_area_px": box_width_px * box_height_px,
            }
        )
    return records


def build_bbox_records(validation_df: pd.DataFrame, class_names: dict[int, str]) -> pd.DataFrame:
    """Build one row per object box from valid, non-empty labels."""
    records: list[dict[str, Any]] = []
    valid_labeled = validation_df[
        (validation_df["status"] == "valid") & (validation_df["num_objects"].astype(int) > 0)
    ]
    for _, row in valid_labeled.iterrows():
        for record in _parse_valid_label_rows(row):
            record["class_name"] = class_names.get(record["class_id"], record["class_name"])
            records.append(record)

    columns = [
        "image_name",
        "class_id",
        "class_name",
        "x_center_norm",
        "y_center_norm",
        "width_norm",
        "height_norm",
        "box_area_norm",
        "image_width",
        "image_height",
        "box_width_px",
        "box_height_px",
        "box_area_px",
    ]
    return pd.DataFrame(records, columns=columns)


def build_input_summary(validation_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact one-row summary for Notebook 01."""
    valid = validation_df[validation_df["status"] == "valid"]
    summary = {
        "total_images": int((validation_df["image_name"].fillna("") != "").sum()),
        "valid_images": int(len(valid)),
        "invalid_rows": int((validation_df["status"] == "invalid").sum()),
        "no_sign_images": int(valid["is_no_sign"].fillna(False).astype(bool).sum()),
        "labeled_images": int((valid["num_objects"].fillna(0).astype(int) > 0).sum()),
        "total_objects": int(valid["num_objects"].fillna(0).astype(int).sum()),
    }
    for class_name in CLASS_COLUMNS.values():
        summary[f"num_{class_name}"] = int(valid[f"num_{class_name}"].fillna(0).astype(int).sum())
    return pd.DataFrame([summary])


def build_class_distribution(validation_df: pd.DataFrame, class_names: dict[int, str]) -> pd.DataFrame:
    """Summarize object counts by class from valid rows."""
    valid = validation_df[validation_df["status"] == "valid"]
    rows = []
    for class_id, fallback_name in CLASS_COLUMNS.items():
        class_name = class_names.get(class_id, fallback_name)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "object_count": int(valid[f"num_{fallback_name}"].fillna(0).astype(int).sum()),
                "image_count": int(valid[f"has_{fallback_name}"].fillna(False).astype(bool).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_split_distribution(split_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize image, label, no-sign, and object counts for each split."""
    rows = []
    for split, group in split_df.groupby("split", sort=True):
        num_images = int(len(group))
        num_no_sign = int(group["is_no_sign"].fillna(False).astype(bool).sum())
        total_objects = int(group["num_objects"].fillna(0).astype(int).sum())
        rows.append(
            {
                "split": split,
                "num_images": num_images,
                "num_labels": int(group["target_label_path"].fillna("").astype(str).ne("").sum()),
                "num_labeled_images": int((group["num_objects"].fillna(0).astype(int) > 0).sum()),
                "num_no_sign_images": num_no_sign,
                "no_sign_ratio": num_no_sign / num_images if num_images else 0.0,
                "total_objects": total_objects,
            }
        )
    return pd.DataFrame(rows)


def summarize_split_class_distribution(split_df: pd.DataFrame, class_names: dict[int, str]) -> pd.DataFrame:
    """Summarize object-level class counts by split."""
    rows = []
    for split, group in split_df.groupby("split", sort=True):
        total_objects = max(int(group["num_objects"].fillna(0).astype(int).sum()), 1)
        for class_id, fallback_name in CLASS_COLUMNS.items():
            count = int(group[f"num_{fallback_name}"].fillna(0).astype(int).sum())
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, fallback_name),
                    "object_count": count,
                    "object_ratio_within_split": count / total_objects,
                }
            )
    return pd.DataFrame(rows)


def summarize_split_image_presence(split_df: pd.DataFrame, class_names: dict[int, str]) -> pd.DataFrame:
    """Summarize image-level class presence by split."""
    rows = []
    for split, group in split_df.groupby("split", sort=True):
        num_images = max(int(len(group)), 1)
        for class_id, fallback_name in CLASS_COLUMNS.items():
            count = int(group[f"has_{fallback_name}"].fillna(False).astype(bool).sum())
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, fallback_name),
                    "images_with_class": count,
                    "image_presence_ratio": count / num_images,
                }
            )
    return pd.DataFrame(rows)


def summarize_split_no_sign_distribution(split_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize no-sign images by split."""
    rows = []
    for split, group in split_df.groupby("split", sort=True):
        num_images = int(len(group))
        no_sign_count = int(group["is_no_sign"].fillna(False).astype(bool).sum())
        rows.append(
            {
                "split": split,
                "num_no_sign_images": no_sign_count,
                "no_sign_ratio": no_sign_count / num_images if num_images else 0.0,
            }
        )
    return pd.DataFrame(rows)


def attach_split_to_bbox_records(bbox_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    """Attach split labels and target image paths to bbox records."""
    if bbox_df.empty:
        return bbox_df.assign(split=pd.Series(dtype="object"), target_image_path=pd.Series(dtype="object"))
    lookup = split_df[["image_name", "split", "target_image_path"]].drop_duplicates("image_name")
    return bbox_df.merge(lookup, on="image_name", how="left")


def summarize_bbox_statistics_by_split(
    bbox_with_split_df: pd.DataFrame,
    class_names: dict[int, str],
) -> pd.DataFrame:
    """Summarize bbox size statistics by split and class."""
    columns = [
        "split",
        "class_id",
        "class_name",
        "num_boxes",
        "mean_box_area_norm",
        "median_box_area_norm",
        "mean_box_width_px",
        "mean_box_height_px",
        "tiny_box_count",
        "tiny_box_ratio",
        "small_box_count",
        "small_box_ratio",
    ]
    if bbox_with_split_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    work = bbox_with_split_df.copy()
    work["is_tiny_box"] = (
        (work["box_area_norm"] < 0.0005)
        | (work["box_width_px"] < 8)
        | (work["box_height_px"] < 8)
    )
    work["is_small_box"] = work["box_area_norm"] < 0.002
    for (split, class_id), group in work.groupby(["split", "class_id"], sort=True):
        num_boxes = int(len(group))
        fallback_name = CLASS_COLUMNS.get(int(class_id), str(class_id))
        tiny_count = int(group["is_tiny_box"].sum())
        small_count = int(group["is_small_box"].sum())
        rows.append(
            {
                "split": split,
                "class_id": int(class_id),
                "class_name": class_names.get(int(class_id), fallback_name),
                "num_boxes": num_boxes,
                "mean_box_area_norm": float(group["box_area_norm"].mean()),
                "median_box_area_norm": float(group["box_area_norm"].median()),
                "mean_box_width_px": float(group["box_width_px"].mean()),
                "mean_box_height_px": float(group["box_height_px"].mean()),
                "tiny_box_count": tiny_count,
                "tiny_box_ratio": tiny_count / num_boxes if num_boxes else 0.0,
                "small_box_count": small_count,
                "small_box_ratio": small_count / num_boxes if num_boxes else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    laplacian = (
        -4 * center
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(laplacian))


def _image_quality_record(row: pd.Series) -> dict[str, Any]:
    image_path = Path(str(row.get("target_image_path") or row.get("source_image_path")))
    record: dict[str, Any] = {
        "split": row["split"],
        "image_name": row["image_name"],
        "image_width": row.get("image_width", None),
        "image_height": row.get("image_height", None),
        "aspect_ratio": row.get("aspect_ratio", None),
        "brightness_mean": None,
        "brightness_std": None,
        "blur_score_laplacian": None,
        "status": "ok",
        "notes": "",
    }
    try:
        with Image.open(image_path) as image:
            gray = np.asarray(image.convert("L"), dtype=np.float32)
            width, height = image.size
        record["image_width"] = int(width)
        record["image_height"] = int(height)
        record["aspect_ratio"] = width / height if height else None
        record["brightness_mean"] = float(gray.mean())
        record["brightness_std"] = float(gray.std())
        record["blur_score_laplacian"] = _laplacian_variance(gray)
    except OSError as exc:
        record["status"] = "failed"
        record["notes"] = str(exc)
    return record


def summarize_image_quality_by_split(split_df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple image quality metrics and summarize them by split."""
    records = [_image_quality_record(row) for _, row in split_df.iterrows()]
    quality_df = pd.DataFrame(records)
    numeric_columns = [
        "image_width",
        "image_height",
        "aspect_ratio",
        "brightness_mean",
        "brightness_std",
        "blur_score_laplacian",
    ]
    if quality_df.empty:
        return pd.DataFrame(columns=["split", "num_images", *[f"mean_{col}" for col in numeric_columns]])

    summary = quality_df.groupby("split", sort=True).agg(
        num_images=("image_name", "count"),
        mean_image_width=("image_width", "mean"),
        mean_image_height=("image_height", "mean"),
        mean_aspect_ratio=("aspect_ratio", "mean"),
        mean_brightness_mean=("brightness_mean", "mean"),
        mean_brightness_std=("brightness_std", "mean"),
        mean_blur_score_laplacian=("blur_score_laplacian", "mean"),
        median_blur_score_laplacian=("blur_score_laplacian", "median"),
    )
    return summary.reset_index()


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_exact_duplicate_leakage(split_df: pd.DataFrame) -> pd.DataFrame:
    """Report exact image-content duplicates that cross split boundaries."""
    rows = []
    for _, row in split_df.iterrows():
        image_path = Path(str(row.get("target_image_path") or row.get("source_image_path")))
        if image_path.exists():
            rows.append(
                {
                    "image_hash": _file_sha1(image_path),
                    "split": row["split"],
                    "image_name": row["image_name"],
                    "image_path": str(image_path),
                }
            )
    hash_df = pd.DataFrame(rows)
    columns = ["image_hash", "splits", "num_images", "image_names", "details"]
    if hash_df.empty:
        return pd.DataFrame(columns=columns)

    leakage_rows = []
    for image_hash, group in hash_df.groupby("image_hash"):
        splits = sorted(group["split"].unique())
        if len(splits) > 1:
            leakage_rows.append(
                {
                    "image_hash": image_hash,
                    "splits": ",".join(splits),
                    "num_images": int(len(group)),
                    "image_names": ";".join(group["image_name"].astype(str)),
                    "details": ";".join(group["image_path"].astype(str)),
                }
            )
    return pd.DataFrame(leakage_rows, columns=columns)


def build_split_warnings(
    split_df: pd.DataFrame,
    bbox_with_split_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build practical warnings that guide split review without stopping the notebook."""
    warnings: list[dict[str, str]] = []
    present_splits = set(split_df["split"].unique()) if not split_df.empty else set()
    for split in ("train", "val", "test"):
        if split not in present_splits:
            warnings.append({"warning_type": "empty_split", "details": f"{split} has no images"})

    for split, group in split_df.groupby("split", sort=True):
        if split == "val" and len(group) < 5:
            warnings.append({"warning_type": "very_small_val_set", "details": f"val has {len(group)} images"})
        if split == "test" and len(group) < 5:
            warnings.append({"warning_type": "very_small_test_set", "details": f"test has {len(group)} images"})
        if int(group["is_no_sign"].fillna(False).astype(bool).sum()) == 0:
            warnings.append({"warning_type": "no_sign_missing_from_split", "details": f"{split} has no no-sign images"})
        if int(group["target_image_path"].map(lambda p: Path(str(p)).exists()).sum()) != int(
            group["target_label_path"].map(lambda p: Path(str(p)).exists()).sum()
        ):
            warnings.append({"warning_type": "image_label_count_mismatch", "details": f"{split} image/label count mismatch"})
        for class_name in CLASS_COLUMNS.values():
            if int(group[f"num_{class_name}"].fillna(0).astype(int).sum()) == 0:
                warnings.append({"warning_type": "class_missing_from_split", "details": f"{split} has no {class_name} objects"})

    if split_df["notes"].astype(str).str.contains("stratification fallback used", regex=False).any():
        warnings.append({"warning_type": "stratification_fallback_used", "details": "split_group_key groups were too small"})

    no_sign = summarize_split_no_sign_distribution(split_df)
    if not no_sign.empty and no_sign["no_sign_ratio"].max() - no_sign["no_sign_ratio"].min() > 0.25:
        warnings.append({"warning_type": "split_distribution_imbalance", "details": "no-sign ratio differs by more than 0.25"})

    duplicates = check_exact_duplicate_leakage(split_df)
    if not duplicates.empty:
        warnings.append({"warning_type": "duplicate_leakage_detected", "details": f"{len(duplicates)} exact duplicate groups cross splits"})

    if bbox_with_split_df is not None and not bbox_with_split_df.empty:
        tiny = (
            (bbox_with_split_df["box_area_norm"] < 0.0005)
            | (bbox_with_split_df["box_width_px"] < 8)
            | (bbox_with_split_df["box_height_px"] < 8)
        )
        tiny_ratio = float(tiny.mean())
        if tiny_ratio > 0.25:
            warnings.append({"warning_type": "tiny_box_high_ratio", "details": f"tiny box ratio is {tiny_ratio:.3f}"})

    return pd.DataFrame(warnings, columns=["warning_type", "details"])
