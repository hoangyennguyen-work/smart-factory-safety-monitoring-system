# Factory Sign Detection

This module trains a lightweight YOLO detector for four ISO-7010 factory safety signs from overhead or elevated factory CCTV footage.

## Fixed Classes

```text
0 = M014_Helmet
1 = M015_Vest
2 = P004_NoThoroughfare
3 = W011_Slippery
```

The detector acts as a backend logic-controller input:

- `M014_Helmet` activates helmet-required logic.
- `M015_Vest` activates vest-required logic.
- `P004_NoThoroughfare` activates restricted-zone logic.
- `W011_Slippery` activates slippery danger-zone logic.

## Data Layout

Place prepared YOLO data directly into:

```text
data/input/images/
data/input/labels/
```

Generated outputs such as splits, augmentations, experiment datasets, reports, runs, and weights are created automatically by notebooks or helper functions when needed.

## Notebook Pipeline

1. `01_validate_and_profile_input_dataset.ipynb`
2. `02_split_and_verify_dataset.ipynb`
3. `03_offline_augmentation.ipynb`
4. `04_build_ablation_datasets.ipynb`
5. `05_candidate_model_training.ipynb`
6. `06_ablation_study.ipynb`
7. `07_final_training_and_evaluation.ipynb`
8. `08_inference_backend_demo.ipynb`

Notebook 02 splits input into train, validation, and test. After that, validation and test must remain untouched.

Geometry-changing augmentation must update labels. Do not copy labels unchanged after rotation, perspective transform, crop, translation, scale, or synthetic placement.

Do not commit raw data, generated data, YOLO runs, model weights, exports, or generated reports.
