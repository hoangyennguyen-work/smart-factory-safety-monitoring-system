# Training Pipeline

Purpose: document candidate triage, ablation training, final training, and final test evaluation for the sign detector.

## Candidate Selection

Notebook 05 performs architecture triage before any ablation study. It trains
all configured lightweight YOLO candidates on `data_exp_D_full_pipeline.yaml`
so every architecture sees the same full-pipeline training condition:
original train data, offline augmented train data, and online augmentation
enabled through Ultralytics training arguments.

Candidate selection uses validation metrics only. Recall is the first sorting
priority because missed factory signs may fail to activate backend safety
rules. The test split is not evaluated or used for candidate selection in this
notebook. The top successful architecture from
`reports/training/candidate_model_ranking.csv` is the architecture Notebook 06
should use for ablation training.

## Ablation Study

## Final Evaluation

## Export
