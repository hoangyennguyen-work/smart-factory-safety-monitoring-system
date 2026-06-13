# Ablation Plan

Purpose: describe A/B/C/D experiment comparisons after fixed train, validation, and test splits exist.

## Experiments

Notebook 04 builds four YOLO dataset folders:

```text
exp_A_original_only
exp_B_online_aug
exp_C_offline_aug
exp_D_full_pipeline
```

Experiment A and B use original training images only. Experiment C and D use original training images plus offline augmented training images.

## Fixed Validation and Test Rule

Only the training set changes between experiments. The validation and test splits are copied identically into all four experiment folders so later comparisons are fair.

Notebook 04 does not train models and does not apply online augmentation. Experiments B and D will enable online augmentation later through training arguments.

## Ranking Metrics

Later notebooks should rank ablation runs using validation metrics only. The test set remains untouched for final evaluation.

## Notebook 06 Ablation Training

Notebook 06 uses the selected architecture from Notebook 05 and trains that
same model across A/B/C/D. It compares training configurations only:
Experiment B and D enable online augmentation, while A and C use the
off/minimal augmentation settings.

Validation metrics decide the best training configuration. The test set is not
used for ablation selection. Notebook 07 will use the selected architecture and
selected ablation experiment for final training and final test evaluation.
