# EDA Plan

Purpose: outline dataset profiling for sign counts, class balance, resolutions, object size, and CCTV-specific issues.

## Notebook 01 Boundary

Notebook 01 performs only pre-split validation and lightweight profiling. Its key output is `reports/profile/input_image_profile.csv`, which records class presence and no-sign status for stratified splitting.

Notebook 01 should not write full EDA outputs to `reports/eda/`.

## Full EDA Starts in Notebook 02

Notebook 02 creates the train/val/test split and then verifies split quality. It should compare:

* class balance across train, val, and test
* no-sign image ratio across train, val, and test
* object counts per split
* bounding-box size and position distributions per split
* image resolution and simple brightness/blur indicators per split
* exact duplicate leakage across split boundaries

Notebook 02 writes split reports under `reports/splits/`, full post-split EDA under `reports/eda/`, and selected figures under `reports/figures/`.

## Model-Selection Rule

The test split may be inspected for split-quality verification and final evaluation only. It must not drive architecture selection, augmentation design, ablation decisions, or threshold tuning.
