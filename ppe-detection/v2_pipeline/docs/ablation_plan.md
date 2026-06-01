# Ablation Plan

The ablation study should answer one question at a time and should only begin after the best candidate architecture has been selected.

## Recommended Order

1. `exp_A_original_only`: no extra augmentation beyond the original split.
2. `exp_B_online_aug`: original split plus online YOLO augmentation.
3. `exp_C_offline_aug`: original split plus offline augmented training data.
4. `exp_D_full_pipeline`: online augmentation plus offline augmentation together.

## Final Test-Set Rule

The test split is reserved for the final chosen configuration only. By default, it comes from `data/test_sources/` and is copied into `data/splits_original/test/` during validation and merge. Do not also reserve 10% of the merged dataset for testing when `use_external_test_set: true`; split the merged dataset in `data/master_original/` into train/validation only. During candidate comparison and ablation work, rely on training/validation metrics and keep the test set untouched.

## Notebook 05 Experiment Datasets

Notebook 05 creates four YOLO-ready dataset folders under `data/experiments/`
so later training and ablation notebooks can compare one factor at a time:

- `exp_A_original_only`: original train split only; online augmentation is off
  during training.
- `exp_B_online_aug`: original train split only; online augmentation is on
  during training.
- `exp_C_offline_aug`: original train split plus offline augmented train
  samples; online augmentation is off or minimal during training.
- `exp_D_full_pipeline`: original train split plus offline augmented train
  samples; online augmentation is on during training.

The validation and test splits are copied identically into every experiment.
Only the training split changes. This keeps validation metrics comparable during
Notebook 07 and preserves the final test set for one-time evaluation of the
selected configuration.

Notebook 05 also writes one Ultralytics dataset YAML file per experiment:

- `data_exp_A_original_only.yaml`
- `data_exp_B_online_aug.yaml`
- `data_exp_C_offline_aug.yaml`
- `data_exp_D_full_pipeline.yaml`

Reports under `reports/experiments/` document copied files, split summaries,
class distributions, and integrity warnings.

## Notebook 07 Ablation Study

Notebook 07 runs the ablation study only after Notebook 06 has selected the
best lightweight architecture. The selected architecture is fixed across all
four experiments so the comparison isolates the training data and augmentation
strategy rather than model capacity.

- `exp_A_original_only`: original training data only; online augmentation off.
- `exp_B_online_aug`: original training data only; online augmentation on.
- `exp_C_offline_aug`: original plus offline augmented training data; online
  augmentation off.
- `exp_D_full_pipeline`: original plus offline augmented training data; online
  augmentation on.

Validation metrics decide the best ablation configuration. The test split stays
untouched until final evaluation, so Notebook 07 must not rank experiments with
test-set results.
