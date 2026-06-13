# AGENTS.md — Factory Sign Detection

## Project Context

This module trains a lightweight real-time YOLO object detection model for ISO-7010 industrial safety signs from overhead / elevated factory CCTV footage.

The sign detector acts as an automated logic controller for the FastAPI backend:

* `M014_Helmet` activates helmet-required PPE logic.
* `M015_Vest` activates high-visibility vest PPE logic.
* `P004_NoThoroughfare` activates restricted-zone logic.
* `W011_Slippery` activates slippery danger-zone logic.

The model must run together with PPE detection and possibly pose detection on one edge device, so recall, inference speed, and model size are critical.

## Class Taxonomy

Train one single multi-class YOLO detector.

The class mapping is fixed and must not change:

```text
0 = M014_Helmet
1 = M015_Vest
2 = P004_NoThoroughfare
3 = W011_Slippery
```

Do not rename, reorder, merge, or split these classes.

## Target Data Domain

The dataset should represent factory CCTV conditions:

* Overhead / elevated camera angles
* Indoor and outdoor industrial areas
* Small signs in the frame
* Wall-mounted, pole-mounted, fence-mounted, and floor-area signs
* Rotation, tilt, perspective distortion, and side views
* Partial occlusion
* Motion blur
* Low resolution / compression artifacts
* Harsh sunlight, shadows, rain, low light, IR / monochrome CCTV
* Hard negatives with similar colors, shapes, helmets, vests, labels, boards, or clothing but no target sign

## Module Structure

Use this structure:

```text
sign-detection/
  AGENTS.md
  README.md

  configs/
    class_names.yaml
    dataset_config.yaml
    augmentation_config.yaml
    training_config.yaml

  data/
    input/
      images/
      labels/

    generated/
      splits_original/
      augmented_train/
      experiments/

  notebooks/
  src/
    dataset/
    analysis/
    augmentation/
    training/
    inference/

  reports/
  runs/
  weights/
  docs/
```

Meaning:

* `data/input/images` and `data/input/labels` are manually provided by the user.
* `data/generated/` is created by the pipeline.
* Do not create `raw_sources`, `test_sources`, or `master_original`.
* Do not manually edit generated folders unless explicitly requested.

## Pipeline

Use this notebook-based pipeline:

```text
01_validate_and_profile_input_dataset.ipynb
02_split_and_verify_dataset.ipynb
03_offline_augmentation.ipynb
04_build_ablation_datasets.ipynb
05_candidate_model_training.ipynb
06_ablation_study.ipynb
07_final_training_and_evaluation.ipynb
08_inference_backend_demo.ipynb
```

Keep notebooks clean and readable. Put reusable logic in `src/`.

## Dataset Rules

The input dataset must follow YOLO format:

```text
data/input/
  images/
  labels/
```

Each image should normally have a matching `.txt` label file.

YOLO label rows must follow:

```text
class_id x_center y_center width height
```

Validation must check:

* image is readable
* image-label pair exists
* label row has exactly 5 values
* class ID is one of `0,1,2,3`
* coordinates are numeric and normalized to `[0,1]`
* width and height are greater than 0
* boxes stay inside image boundaries within small tolerance

Empty labels are allowed only when the image is intentionally treated as a hard negative. Otherwise, empty labels should be reported clearly.

## Split Strategy

The dataset is split from `data/input/` into:

```text
data/generated/splits_original/train/
data/generated/splits_original/val/
data/generated/splits_original/test/
```

Default split:

```text
train = 70%
val   = 20%
test  = 10%
```

The split should be deterministic using a fixed random seed.

Prefer stratified splitting by image-level class presence:

```text
M014_Helmet present / absent
M015_Vest present / absent
P004_NoThoroughfare present / absent
W011_Slippery present / absent
hard negative / not hard negative
```

After splitting, verify that train, validation, and test have similar class distribution. The test set may be inspected only for split-quality verification and final evaluation.

Do not use the test set for architecture selection, augmentation design, hyperparameter tuning, confidence threshold tuning, or ablation decisions.

## Augmentation Strategy

Sign detection needs stronger geometric augmentation than PPE detection.

Offline augmentation must be generated from training images only. Validation and test sets must remain untouched.

Recommended offline augmentation groups:

```text
geometric:
  rotation, tilt, perspective, translation, scale, crop, distant CCTV simulation

photometric:
  IR / grayscale, harsh sunlight, low light, shadow, contrast, brightness

weather_quality:
  motion blur, Gaussian blur, JPEG compression, low resolution, sensor noise, rain / dirty lens

synthetic_placement:
  optional sign copy-paste or synthetic placement only when labels are updated correctly
```

Important label rule:

* Geometry-changing transforms must update bounding boxes.
* Do not copy labels unchanged after rotation, perspective transform, crop, scale, or synthetic placement.
* Labels may be copied unchanged only for non-geometric transforms such as grayscale, brightness, contrast, blur, compression, noise, shadow, and low-light.

Avoid extreme transformations that make signs unreadable or unrealistic for factory CCTV.

## Online Augmentation

Use Ultralytics online augmentation during training.

For sign detection, moderate geometry augmentation is allowed:

* mild rotation
* translation
* scale
* perspective
* HSV
* mosaic
* mixup if still visually realistic

Avoid vertical flip unless there is a strong reason. Signs upside down are usually unrealistic for factory CCTV.

## Candidate Architectures

Candidate triage should support:

```text
yolov8n.pt
yolov9t.pt
yolov10n.pt
yolo11n.pt
yolo12n.pt
yolo26n.pt
```

If a model is unsupported by the installed Ultralytics version or weights are unavailable, skip it safely and report the failure.

Do not write raw PyTorch training loops unless explicitly requested. Use Ultralytics `YOLO(...)`.

## Model Selection Priority

Select the final architecture using validation metrics only.

Priority:

```text
1. Recall
2. mAP50
3. mAP50-95
4. Inference latency / FPS
5. Parameter size / VRAM footprint
```

Recall is prioritized because missed signs may fail to activate backend safety rules.

## Ablation Experiments

Use fixed validation and test sets across all experiments.

```text
A_original_only:
  train = original train only
  online augmentation = off

B_online_aug:
  train = original train only
  online augmentation = on

C_offline_aug:
  train = original train + offline augmented train
  online augmentation = off/minimal

D_full_pipeline:
  train = original train + offline augmented train
  online augmentation = on
```

Only the training condition changes. Validation and test sets must remain identical.

## Final Evaluation Rule

The test set is used only once for final model evaluation.

Do not change the following based on test results:

* architecture
* dataset choice
* augmentation settings
* confidence threshold
* backend rule logic

Any analysis after final test evaluation must be clearly marked as post-hoc.

## Backend Integration Context

Expected downstream backend behavior:

```text
M014_Helmet detected          -> activate helmet-required rule
M015_Vest detected            -> activate vest-required rule
P004_NoThoroughfare detected  -> activate restricted-zone rule
W011_Slippery detected        -> activate slippery danger-zone rule
sign disappears / expires     -> deactivate or timeout related rule
```

The final model should support real-time inference and export to deployment formats such as ONNX.

## Tech Stack

Use:

* Python 3.10+
* Anaconda environment
* Jupyter notebooks
* Ultralytics YOLO
* OpenCV / Pillow
* pandas / NumPy
* Matplotlib / Seaborn
* PyYAML

## Git Hygiene

Do not commit:

* raw input images or labels
* generated split datasets
* augmented datasets
* experiment datasets
* YOLO runs
* model weights
* exported models
* generated reports / figures

Commit only:

* source code
* notebooks
* configs
* documentation
* `.gitkeep` placeholders

## Coding Rules

* Use `pathlib.Path`
* Use type hints
* Avoid hardcoded absolute paths
* Copy files instead of moving them
* Never silently overwrite data
* Fail gracefully with clear messages
* Keep outputs deterministic using fixed random seeds
* Keep reusable logic in `src/`
* Keep notebooks as orchestration only
* Do not modify the PPE pipeline unless explicitly requested

## Testing Before Final Response

After code changes, run relevant checks:

```powershell
python -m compileall sign-detection/src
```

Also:

* Parse YAML configs
* Validate notebook JSON
* Check imports
* Run tiny synthetic dataset tests when practical
* Finish with:

```powershell
git status --short
```

Final Codex response should summarize:

1. What changed
2. Files created or modified
3. Tests/checks run
4. Any warnings or skipped items
5. What to do next
