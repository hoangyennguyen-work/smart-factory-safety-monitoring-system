"""Experiment dataset builders for factory sign detection."""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_COLUMNS: dict[int, str] = {
    0: "M014_Helmet",
    1: "M015_Vest",
    2: "P004_NoThoroughfare",
    3: "W011_Slippery",
}
EXPERIMENTS = {
    "exp_A_original_only": {"include_offline_aug": False, "online_aug_later": False},
    "exp_B_online_aug": {"include_offline_aug": False, "online_aug_later": True},
    "exp_C_offline_aug": {"include_offline_aug": True, "online_aug_later": False},
    "exp_D_full_pipeline": {"include_offline_aug": True, "online_aug_later": True},
}
AUGMENTED_SOURCE_TYPES = {
    "geometric": "augmented_geometric",
    "photometric": "augmented_photometric",
    "weather_quality": "augmented_weather_quality",
    "synthetic_placement": "augmented_synthetic_placement",
}


def collect_yolo_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Collect sorted image-label pairs from YOLO image and label folders."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    pairs: list[tuple[Path, Path]] = []
    if not images_dir.exists() or not labels_dir.exists():
        return pairs
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def _read_label_counts(label_path: Path) -> tuple[int, Counter[int]]:
    text = Path(label_path).read_text(encoding="utf-8").strip()
    counts: Counter[int] = Counter()
    if not text:
        return 0, counts
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 1:
            continue
        counts[int(float(parts[0]))] += 1
    return sum(counts.values()), counts


def _prepare_experiment_root(experiments_dir: Path, overwrite: bool) -> None:
    experiments_dir = Path(experiments_dir)
    if experiments_dir.exists():
        existing_files = [path for path in experiments_dir.rglob("*") if path.is_file()]
        if existing_files and not overwrite:
            raise FileExistsError(
                f"Experiment output folder {experiments_dir} already contains files. "
                "Set overwrite=True to rebuild ablation datasets."
            )
        if overwrite:
            for experiment in EXPERIMENTS:
                experiment_dir = experiments_dir / experiment
                if experiment_dir.exists():
                    shutil.rmtree(experiment_dir)

    for experiment in EXPERIMENTS:
        for split in ("train", "val", "test"):
            (experiments_dir / experiment / split / "images").mkdir(parents=True, exist_ok=True)
            (experiments_dir / experiment / split / "labels").mkdir(parents=True, exist_ok=True)


def _safe_target_paths(
    image_path: Path,
    label_path: Path,
    target_images_dir: Path,
    target_labels_dir: Path,
) -> tuple[Path, Path, bool]:
    target_image = target_images_dir / image_path.name
    target_label = target_labels_dir / label_path.name
    conflict = False
    if not target_image.exists() and not target_label.exists():
        return target_image, target_label, conflict

    conflict = True
    for index in range(1, 1000):
        suffix = f"_dup{index:03d}"
        candidate_image = target_images_dir / f"{image_path.stem}{suffix}{image_path.suffix}"
        candidate_label = target_labels_dir / f"{label_path.stem}{suffix}{label_path.suffix}"
        if not candidate_image.exists() and not candidate_label.exists():
            return candidate_image, candidate_label, conflict
    raise RuntimeError(f"Could not create a conflict-safe filename for {image_path.name}")


def copy_yolo_pairs(
    pairs: Iterable[tuple[Path, Path]],
    target_split_dir: Path,
    experiment: str,
    split: str,
    source_type: str,
) -> list[dict[str, str]]:
    """Copy YOLO image-label pairs into one experiment split folder."""
    target_images_dir = Path(target_split_dir) / "images"
    target_labels_dir = Path(target_split_dir) / "labels"
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for image_path, label_path in pairs:
        target_image, target_label, conflict = _safe_target_paths(
            image_path, label_path, target_images_dir, target_labels_dir
        )
        notes = "filename conflict resolved with suffix" if conflict else ""
        try:
            # Copy files instead of moving so original splits and augmentation outputs remain untouched.
            shutil.copy2(image_path, target_image)
            shutil.copy2(label_path, target_label)
            status = "copied"
        except OSError as exc:
            status = "failed"
            notes = f"{notes}; {exc}".strip("; ")

        rows.append(
            {
                "experiment": experiment,
                "split": split,
                "source_type": source_type,
                "original_image_path": str(image_path),
                "original_label_path": str(label_path),
                "copied_image_path": str(target_image),
                "copied_label_path": str(target_label),
                "status": status,
                "notes": notes,
            }
        )
    return rows


def write_yolo_dataset_yaml(
    experiment: str,
    output_yaml_dir: Path,
    class_names: dict[int, str],
) -> Path:
    """Write one Ultralytics dataset YAML for an ablation experiment."""
    output_yaml_dir = Path(output_yaml_dir)
    output_yaml_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_yaml_dir / f"data_{experiment}.yaml"
    data = {
        "path": f"data/generated/experiments/{experiment}",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": dict(sorted(class_names.items())),
    }
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    return yaml_path


def summarize_experiment_dataset(experiments_dir: Path, experiment: str) -> pd.DataFrame:
    """Summarize image, label, no-sign, and object counts for one experiment."""
    rows = []
    for split in ("train", "val", "test"):
        split_dir = Path(experiments_dir) / experiment / split
        image_dir = split_dir / "images"
        label_dir = split_dir / "labels"
        images = [path for path in image_dir.iterdir()] if image_dir.exists() else []
        labels = [path for path in label_dir.iterdir()] if label_dir.exists() else []
        label_by_stem = {path.stem: path for path in labels if path.is_file()}
        total_objects = 0
        no_sign_images = 0
        class_counts: Counter[int] = Counter()
        for image_path in images:
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_by_stem.get(image_path.stem)
            if label_path is None:
                continue
            object_count, counts = _read_label_counts(label_path)
            total_objects += object_count
            class_counts.update(counts)
            if object_count == 0:
                no_sign_images += 1
        row = {
            "experiment": experiment,
            "split": split,
            "num_images": int(len([p for p in images if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])),
            "num_labels": int(len([p for p in labels if p.is_file() and p.suffix.lower() == ".txt"])),
            "num_no_sign_images": int(no_sign_images),
            "total_objects": int(total_objects),
            "notes": "online augmentation handled later during training",
        }
        for class_id, class_name in CLASS_COLUMNS.items():
            row[f"num_{class_name}"] = int(class_counts[class_id])
        rows.append(row)
    return pd.DataFrame(rows)


def _split_filenames(experiments_dir: Path, experiment: str, split: str) -> set[str]:
    image_dir = Path(experiments_dir) / experiment / split / "images"
    if not image_dir.exists():
        return set()
    return {path.name for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}


def verify_experiment_integrity(
    report_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    experiments_dir: Path,
) -> pd.DataFrame:
    """Build integrity warnings for generated ablation datasets."""
    warnings: list[dict[str, str]] = []
    for _, row in summary_df.iterrows():
        experiment = str(row["experiment"])
        split = str(row["split"])
        if int(row["num_images"]) == 0:
            warnings.append({"experiment": experiment, "split": split, "warning_type": "empty_split", "details": "split has no images"})
        if int(row["num_images"]) != int(row["num_labels"]):
            warnings.append({"experiment": experiment, "split": split, "warning_type": "image_label_count_mismatch", "details": f"{row['num_images']} images vs {row['num_labels']} labels"})
        for class_name in CLASS_COLUMNS.values():
            if int(row[f"num_{class_name}"]) == 0:
                warnings.append({"experiment": experiment, "split": split, "warning_type": "class_missing_from_split", "details": f"no {class_name} objects"})

    for _, row in report_df.iterrows():
        experiment = str(row["experiment"])
        split = str(row["split"])
        if "filename conflict" in str(row["notes"]):
            warnings.append({"experiment": experiment, "split": split, "warning_type": "filename_conflict", "details": str(row["copied_image_path"])})
        if row["status"] != "copied":
            warnings.append({"experiment": experiment, "split": split, "warning_type": "missing_image", "details": str(row["original_image_path"])})

    experiments = list(EXPERIMENTS)
    base_val = _split_filenames(experiments_dir, experiments[0], "val")
    base_test = _split_filenames(experiments_dir, experiments[0], "test")
    for experiment in experiments[1:]:
        if _split_filenames(experiments_dir, experiment, "val") != base_val:
            warnings.append({"experiment": experiment, "split": "val", "warning_type": "val_set_differs_across_experiments", "details": "validation image names differ from exp_A_original_only"})
        if _split_filenames(experiments_dir, experiment, "test") != base_test:
            warnings.append({"experiment": experiment, "split": "test", "warning_type": "test_set_differs_across_experiments", "details": "test image names differ from exp_A_original_only"})
    return pd.DataFrame(warnings, columns=["experiment", "split", "warning_type", "details"])


def _collect_augmented_pairs(augmented_train_dir: Path) -> dict[str, list[tuple[Path, Path]]]:
    augmented: dict[str, list[tuple[Path, Path]]] = {}
    for folder_name, source_type in AUGMENTED_SOURCE_TYPES.items():
        pairs = collect_yolo_pairs(
            Path(augmented_train_dir) / folder_name / "images",
            Path(augmented_train_dir) / folder_name / "labels",
        )
        augmented[source_type] = pairs
    return augmented


def build_ablation_datasets(
    splits_original_dir: Path,
    augmented_train_dir: Path,
    experiments_dir: Path,
    class_names: dict[int, str],
    output_yaml_dir: Path,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build A/B/C/D ablation YOLO datasets and write dataset YAML files."""
    splits_original_dir = Path(splits_original_dir)
    augmented_train_dir = Path(augmented_train_dir)
    experiments_dir = Path(experiments_dir)
    output_yaml_dir = Path(output_yaml_dir)

    original_pairs = {
        "train": collect_yolo_pairs(splits_original_dir / "train" / "images", splits_original_dir / "train" / "labels"),
        "val": collect_yolo_pairs(splits_original_dir / "val" / "images", splits_original_dir / "val" / "labels"),
        "test": collect_yolo_pairs(splits_original_dir / "test" / "images", splits_original_dir / "test" / "labels"),
    }
    missing = [split for split, pairs in original_pairs.items() if not pairs]
    if missing:
        raise ValueError(f"Missing original split pairs for: {', '.join(missing)}")

    augmented_pairs = _collect_augmented_pairs(augmented_train_dir)
    augmented_total = sum(len(pairs) for pairs in augmented_pairs.values())

    _prepare_experiment_root(experiments_dir, overwrite=overwrite)

    report_rows: list[dict[str, str]] = []
    for experiment, config in EXPERIMENTS.items():
        experiment_dir = experiments_dir / experiment
        report_rows.extend(copy_yolo_pairs(original_pairs["train"], experiment_dir / "train", experiment, "train", "original_train"))
        if config["include_offline_aug"]:
            for source_type, pairs in augmented_pairs.items():
                report_rows.extend(copy_yolo_pairs(pairs, experiment_dir / "train", experiment, "train", source_type))
        report_rows.extend(copy_yolo_pairs(original_pairs["val"], experiment_dir / "val", experiment, "val", "original_val"))
        report_rows.extend(copy_yolo_pairs(original_pairs["test"], experiment_dir / "test", experiment, "test", "original_test"))
        write_yolo_dataset_yaml(experiment, output_yaml_dir, class_names)

    report_df = pd.DataFrame(
        report_rows,
        columns=[
            "experiment",
            "split",
            "source_type",
            "original_image_path",
            "original_label_path",
            "copied_image_path",
            "copied_label_path",
            "status",
            "notes",
        ],
    )
    summary_df = pd.concat(
        [summarize_experiment_dataset(experiments_dir, experiment) for experiment in EXPERIMENTS],
        ignore_index=True,
    )
    warnings_df = verify_experiment_integrity(report_df, summary_df, experiments_dir)
    if augmented_total == 0:
        extra = pd.DataFrame(
            [
                {
                    "experiment": "exp_C_offline_aug",
                    "split": "train",
                    "warning_type": "offline_augmented_data_missing",
                    "details": f"no augmented pairs found under {augmented_train_dir}",
                },
                {
                    "experiment": "exp_D_full_pipeline",
                    "split": "train",
                    "warning_type": "offline_augmented_data_missing",
                    "details": f"no augmented pairs found under {augmented_train_dir}",
                },
            ]
        )
        warnings_df = pd.concat([warnings_df, extra], ignore_index=True)
    return report_df, summary_df, warnings_df
