"""Split PPE v2 input source lanes into generated YOLO train/val/test folders.

The active v2 split policy is source-aware:

``train``
    Open-source data plus the training portion of factory-source data.
``val``
    Factory-source validation portion only.
``test``
    Test-source data only.

Factory train/validation splitting is stratified by the PPE-role classes that
matter most for downstream compliance logic: helmet, vest, and
cleaning_coverall. Person is still copied and counted, but it is not part of
the stratification key because nearly every useful PPE scene contains people.
"""

from __future__ import annotations

import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = ("train", "val", "test")
CLASS_COUNT_COLUMNS = {
    0: "num_person",
    1: "num_helmet",
    2: "num_vest",
    3: "num_cleaning_coverall",
}
STRATIFY_CLASS_IDS = (1, 2, 3)


@dataclass(frozen=True, slots=True)
class YoloPair:
    """One validated image-label pair plus label-derived split metadata."""

    source: str
    image_path: Path
    label_path: Path
    class_counts: dict[int, int]

    @property
    def base_name(self) -> str:
        return self.image_path.stem

    @property
    def has_helmet(self) -> bool:
        return self.class_counts.get(1, 0) > 0

    @property
    def has_vest(self) -> bool:
        return self.class_counts.get(2, 0) > 0

    @property
    def has_cleaning_coverall(self) -> bool:
        return self.class_counts.get(3, 0) > 0

    @property
    def stratify_key(self) -> str:
        """Presence key used to balance factory train/validation splits."""
        return (
            f"helmet={int(self.has_helmet)}|"
            f"vest={int(self.has_vest)}|"
            f"coverall={int(self.has_cleaning_coverall)}"
        )


def split_input_sources(
    open_source_dir: Path,
    factory_source_dir: Path,
    test_source_dir: Path,
    output_splits_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the PPE v2 train/val/test split from source lanes.

    Args:
        open_source_dir: Input root containing open-source ``images/`` and
            ``labels/``. These samples are copied to train only.
        factory_source_dir: Input root containing factory ``images/`` and
            ``labels/``. These samples are split into train and validation.
        test_source_dir: Input root containing final test ``images/`` and
            ``labels/``. These samples are copied to test only.
        output_splits_dir: Generated output root for ``train/``, ``val/``, and
            ``test/`` YOLO folders.
        train_ratio: Factory-source training ratio.
        val_ratio: Factory-source validation ratio.
        seed: Deterministic shuffle seed.
        overwrite: If ``True``, existing split files are cleared first.

    Returns:
        ``(copy_report, split_summary, split_warnings)`` as pandas DataFrames.

    Raises:
        ValueError: If ratios are invalid or required image-label pairs are
            mismatched.
        FileExistsError: If output split folders already contain files and
            ``overwrite`` is ``False``.
    """
    _validate_train_val_ratios(train_ratio, val_ratio)
    output_splits_dir = Path(output_splits_dir)
    for split_name in SPLIT_NAMES:
        _ensure_split_dirs(output_splits_dir, split_name)
    _prepare_output_folders(output_splits_dir, overwrite=overwrite)

    open_pairs = collect_yolo_pairs(open_source_dir, "open_source", required=False)
    factory_pairs = collect_yolo_pairs(factory_source_dir, "factory_source", required=True)
    test_pairs = collect_yolo_pairs(test_source_dir, "test_source", required=False)

    factory_train_pairs, factory_val_pairs, stratification_warnings = (
        stratified_factory_train_val_split(
            factory_pairs=factory_pairs,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    )

    split_plan = {
        "train": [*open_pairs, *factory_train_pairs],
        "val": factory_val_pairs,
        "test": test_pairs,
    }

    copy_rows: list[dict[str, Any]] = []
    used_targets: dict[str, set[str]] = {split_name: set() for split_name in SPLIT_NAMES}
    for split_name, pairs in split_plan.items():
        for pair in pairs:
            copy_rows.append(
                copy_pair_to_split(
                    pair=pair,
                    output_splits_dir=output_splits_dir,
                    split_name=split_name,
                    used_names=used_targets[split_name],
                )
            )

    copy_report = pd.DataFrame(copy_rows)
    split_summary = summarize_split_assignments(copy_report)
    split_warnings = build_split_warnings(
        copy_report=copy_report,
        split_summary=split_summary,
        stratification_warnings=stratification_warnings,
    )
    return copy_report, split_summary, split_warnings


def collect_yolo_pairs(
    source_dir: Path,
    source_name: str,
    required: bool,
) -> list[YoloPair]:
    """Collect matching YOLO image-label pairs from one source lane.

    The function expects a direct source layout:

    ``source_dir/images/<name>.jpg``
    ``source_dir/labels/<name>.txt``

    It parses label files only for class counts and split stratification; full
    label validation remains Notebook 01's job.
    """
    source_dir = Path(source_dir)
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        if required:
            raise FileNotFoundError(
                f"{source_name} must contain images/ and labels/: {source_dir}"
            )
        return []

    image_paths_by_stem: dict[str, list[Path]] = {}
    for image_path in sorted(images_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
            image_paths_by_stem.setdefault(image_path.stem, []).append(image_path)

    duplicate_stems = sorted(
        stem for stem, paths in image_paths_by_stem.items() if len(paths) > 1
    )
    if duplicate_stems:
        raise ValueError(
            f"Duplicate image stems in {source_name}: {', '.join(duplicate_stems[:10])}"
        )

    label_paths_by_stem = {
        label_path.stem: label_path
        for label_path in sorted(labels_dir.iterdir())
        if label_path.is_file() and label_path.suffix.lower() == ".txt"
    }

    image_stems = set(image_paths_by_stem)
    label_stems = set(label_paths_by_stem)
    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)
    if missing_labels or missing_images:
        messages: list[str] = []
        if missing_labels:
            messages.append("images without labels: " + ", ".join(missing_labels[:10]))
        if missing_images:
            messages.append("labels without images: " + ", ".join(missing_images[:10]))
        raise ValueError(f"{source_name} image-label mismatch; " + "; ".join(messages))

    return [
        YoloPair(
            source=source_name,
            image_path=image_paths_by_stem[stem][0],
            label_path=label_paths_by_stem[stem],
            class_counts=_count_label_classes(label_paths_by_stem[stem]),
        )
        for stem in sorted(image_stems)
    ]


def stratified_factory_train_val_split(
    factory_pairs: list[YoloPair],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[YoloPair], list[YoloPair], list[dict[str, str]]]:
    """Split factory pairs while preserving helmet/vest/coverall presence.

    A simple group-wise stratification is enough here. Each factory image is
    grouped by whether it contains helmet, vest, and cleaning_coverall. Groups
    with at least two samples contribute at least one validation image; singleton
    groups stay in train and produce a warning because they cannot be balanced.
    """
    grouped_pairs: dict[str, list[YoloPair]] = defaultdict(list)
    for pair in factory_pairs:
        grouped_pairs[pair.stratify_key].append(pair)

    rng = random.Random(seed)
    train_pairs: list[YoloPair] = []
    val_pairs: list[YoloPair] = []
    warnings: list[dict[str, str]] = []

    for stratify_key in sorted(grouped_pairs):
        group_pairs = grouped_pairs[stratify_key].copy()
        rng.shuffle(group_pairs)
        if len(group_pairs) == 1:
            train_pairs.extend(group_pairs)
            warnings.append(
                {
                    "warning_type": "stratification_singleton",
                    "details": (
                        f"{stratify_key} has one factory image, so it cannot "
                        "be represented in both train and val"
                    ),
                }
            )
            continue

        val_count = max(1, round(len(group_pairs) * val_ratio))
        val_count = min(val_count, len(group_pairs) - 1)
        val_pairs.extend(group_pairs[:val_count])
        train_pairs.extend(group_pairs[val_count:])

    # Shuffle final split lists so each split is not ordered by stratification
    # group. This remains deterministic because the same rng is reused.
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    return train_pairs, val_pairs, warnings


def copy_pair_to_split(
    pair: YoloPair,
    output_splits_dir: Path,
    split_name: str,
    used_names: set[str],
) -> dict[str, Any]:
    """Copy one image-label pair into a split folder with a safe filename.

    Source prefixes make train-set filename collisions deterministic and easy to
    trace, especially because train combines open-source and factory data.
    """
    images_dir = Path(output_splits_dir) / split_name / "images"
    labels_dir = Path(output_splits_dir) / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    target_stem = _safe_target_stem(
        preferred_stem=f"{pair.source}__{pair.base_name}",
        used_names=used_names,
    )
    target_image_path = images_dir / f"{target_stem}{pair.image_path.suffix.lower()}"
    target_label_path = labels_dir / f"{target_stem}.txt"

    shutil.copy2(pair.image_path, target_image_path)
    shutil.copy2(pair.label_path, target_label_path)

    return {
        "split": split_name,
        "source": pair.source,
        "stratify_key": pair.stratify_key,
        "original_image_path": str(pair.image_path),
        "original_label_path": str(pair.label_path),
        "copied_image_path": str(target_image_path),
        "copied_label_path": str(target_label_path),
        "status": "copied",
        "notes": "",
        "num_objects": sum(pair.class_counts.values()),
        **{
            column: int(pair.class_counts.get(class_id, 0))
            for class_id, column in CLASS_COUNT_COLUMNS.items()
        },
    }


def summarize_split_assignments(copy_report: pd.DataFrame) -> pd.DataFrame:
    """Create compact split counts from the copy report."""
    columns = [
        "split",
        "num_images",
        "num_labels",
        "num_objects",
        *CLASS_COUNT_COLUMNS.values(),
        "sources",
    ]
    if copy_report.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for split_name in SPLIT_NAMES:
        split_df = copy_report.loc[copy_report["split"].eq(split_name)]
        rows.append(
            {
                "split": split_name,
                "num_images": int(split_df["copied_image_path"].nunique()),
                "num_labels": int(split_df["copied_label_path"].nunique()),
                "num_objects": int(split_df["num_objects"].sum()),
                **{
                    column: int(split_df[column].sum()) if column in split_df else 0
                    for column in CLASS_COUNT_COLUMNS.values()
                },
                "sources": ", ".join(sorted(split_df["source"].unique()))
                if not split_df.empty
                else "",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_split_warnings(
    copy_report: pd.DataFrame,
    split_summary: pd.DataFrame,
    stratification_warnings: list[dict[str, str]],
) -> pd.DataFrame:
    """Build split warnings for user review without stopping the notebook."""
    rows = list(stratification_warnings)

    for row in split_summary.itertuples(index=False):
        if int(row.num_images) == 0:
            rows.append(
                {
                    "warning_type": "empty_split",
                    "details": f"{row.split} has no images",
                }
            )

    for class_column in ("num_helmet", "num_vest", "num_cleaning_coverall"):
        for split_name in ("train", "val"):
            count = int(
                split_summary.loc[
                    split_summary["split"].eq(split_name),
                    class_column,
                ].sum()
            )
            if count == 0:
                rows.append(
                    {
                        "warning_type": "class_missing_from_split",
                        "details": f"{class_column} is missing from {split_name}",
                    }
                )

    if not copy_report.empty:
        val_sources = set(copy_report.loc[copy_report["split"].eq("val"), "source"])
        if val_sources - {"factory_source"}:
            rows.append(
                {
                    "warning_type": "unexpected_validation_source",
                    "details": f"Validation contains non-factory sources: {sorted(val_sources)}",
                }
            )

    return pd.DataFrame(rows, columns=["warning_type", "details"])


def _count_label_classes(label_path: Path) -> dict[int, int]:
    """Count class IDs in one YOLO label file."""
    counts = {class_id: 0 for class_id in CLASS_COUNT_COLUMNS}
    raw_text = Path(label_path).read_text(encoding="utf-8").strip()
    if not raw_text:
        return counts
    for line in raw_text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if class_id in counts:
            counts[class_id] += 1
    return counts


def _safe_target_stem(preferred_stem: str, used_names: set[str]) -> str:
    """Return a deterministic unused target stem."""
    if preferred_stem not in used_names:
        used_names.add(preferred_stem)
        return preferred_stem

    duplicate_index = 1
    while True:
        candidate = f"{preferred_stem}__dup{duplicate_index:03d}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        duplicate_index += 1


def _validate_train_val_ratios(train_ratio: float, val_ratio: float) -> None:
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("Factory train and validation ratios must both be positive")
    total = train_ratio + val_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Factory split.train + split.val must equal 1.0")


def _prepare_output_folders(output_dir: Path, overwrite: bool) -> None:
    occupied_splits = [
        split_name for split_name in SPLIT_NAMES if _split_has_files(output_dir, split_name)
    ]
    if occupied_splits and not overwrite:
        raise FileExistsError(
            "Split folder(s) already contain files: "
            f"{', '.join(occupied_splits)}. Set overwrite=True to regenerate."
        )
    if overwrite:
        for split_name in SPLIT_NAMES:
            clear_split_folder(output_dir, split_name)


def clear_split_folder(output_dir: Path, split_name: str) -> None:
    """Remove files from one generated split's image and label folders."""
    for child_dir in _split_dirs(Path(output_dir), split_name):
        if not child_dir.exists():
            continue
        for file_path in child_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()


def _ensure_split_dirs(output_dir: Path, split_name: str) -> None:
    for child_dir in _split_dirs(output_dir, split_name):
        child_dir.mkdir(parents=True, exist_ok=True)


def _split_dirs(output_dir: Path, split_name: str) -> tuple[Path, Path]:
    split_dir = Path(output_dir) / split_name
    return split_dir / "images", split_dir / "labels"


def _split_has_files(output_dir: Path, split_name: str) -> bool:
    return any(
        file_path.is_file()
        for child_dir in _split_dirs(output_dir, split_name)
        if child_dir.exists()
        for file_path in child_dir.iterdir()
    )
