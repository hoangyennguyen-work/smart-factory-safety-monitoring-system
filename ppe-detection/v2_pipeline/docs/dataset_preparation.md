# Dataset Preparation

The PPE v2 pipeline now uses direct input source lanes instead of a merged
`master_original` folder.

## Input Layout

Place local YOLO data under:

```text
data/input/open_source/
  images/
  labels/

data/input/factory_source/
  images/
  labels/

data/input/test_source/
  images/
  labels/
```

Source meaning:

- `open_source`: public or external data used for training only.
- `factory_source`: target-domain CCTV data used for training and validation.
- `test_source`: final untouched test data used only for final evaluation.

Each image should have a matching `.txt` label file with the same base name.

## Class Map

Use the fixed four-class YOLO schema:

```text
0 = person
1 = helmet
2 = vest
3 = cleaning_coverall
```

Do not add violation classes. Role and violation logic is handled after
detection by the backend.

## Notebook 01

`01_validate_and_merge_dataset.ipynb` now validates the three input source lanes
only. It does not merge, split, augment, train, or modify input files.

Validation checks include:

- readable image
- supported image extension
- matching image-label pair
- valid class ID
- five YOLO values per non-empty label row
- numeric normalized coordinates
- positive box width and height
- box boundaries inside the image
- duplicate base-name and duplicate-content warnings

Reports are written under `reports/validation/`:

- `validation_report.csv`
- `source_summary.csv`

Invalid rows and warnings are displayed in the notebook by filtering
`validation_report.csv`; no separate invalid-samples report is needed.

## Split Policy

The later split notebook should use:

- `train`: valid `open_source` samples plus the training portion of
  `factory_source`.
- `val`: validation portion of `factory_source` only.
- `test`: valid `test_source` only.

This keeps validation and test focused on the real factory domain while still
using open-source data to improve training coverage.

Notebook `03_split_dataset.ipynb` creates this generated split under:

```text
data/generated/splits/train/
data/generated/splits/val/
data/generated/splits/test/
```

The factory train/validation split is stratified by whether images contain:

- `helmet`
- `vest`
- `cleaning_coverall`

If a class or class-combination appears in too few factory images to be present
in both train and validation, Notebook 03 records a warning rather than hiding
the imbalance.

## Generated Data

Generated data belongs under:

```text
data/generated/splits/
data/generated/augmented/
data/generated/experiments/
```

Do not recreate `data/master_original/`, `data/raw_sources/`, or old
train-only `data/open_source_train/` outputs for the new pipeline.
