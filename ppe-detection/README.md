# PPE Detection Module — Preprocessing Handover Guide

## 1. Project Overview
This folder contains the **EDA and preprocessing pipeline** for the Smart Factory Safety Monitoring System's PPE dataset. 
*Note: This repository currently handles data preparation only. It does not contain model training code.*

## 2. Folder Structure
*   `data/CHV_dataset/` — The raw original dataset (**do not modify**).
*   `data/CHV_yolo/` — The final, clean, processed YOLOv8 dataset ready for training.
*   `notebooks/01_eda.ipynb` — Initial Exploratory Data Analysis on the raw dataset.
*   `notebooks/02_preprocessing.ipynb` — The core preprocessing, cleaning, and formatting pipeline.
*   `reports/` — Generated summaries, plots, validation logs, and distributions.

## 3. Environment Setup
To ensure reproducibility and high-performance processing, please set up a clean Anaconda environment using the project's root `environment.yml`:

```bash
# Create the environment from root configuration
conda env create -f ../environment.yml

# Activate the environment
conda activate smart-factory-safety

# Register the Jupyter kernel
python -m ipykernel install --user --name=smart-factory-safety --display-name "Python (smart-factory-safety)"
```

For a comprehensive walkthrough and alternative setup options, please refer to the main [ANACONDA_SETUP_GUIDE.md](../ANACONDA_SETUP_GUIDE.md).

## 4. How to Run the Notebooks
If you need to regenerate the processed dataset or reports from scratch, follow this exact order:
1. Open `notebooks/01_eda.ipynb`.
2. **Run all cells** to execute the raw dataset EDA.
3. Open `notebooks/02_preprocessing.ipynb`.
4. **Run all cells** to fully regenerate the `data/CHV_yolo` dataset and all metric reports.

## 5. Final Dataset for Model Training
If you are here to train the YOLOv8 model, the dataset is already fully prepared! You should point your training script to:
**`data/CHV_yolo/data.yaml`**

## 6. Final Classes
The original 6 color-specific helmet classes have been streamlined into 3 essential classes for YOLOv8:
*   **0:** person
*   **1:** helmet
*   **2:** vest

## 7. Important Design Decision
*   **No absence classes:** We explicitly did not create `no_helmet` or `no_vest` target classes. 
*   **Why?** The object detection model's only job is to find physical gear. Safety violations (e.g., determining if a detected person is missing a helmet) will be calculated later using backend spatial logic (bounding box intersection algorithms).

## 8. Generated Reports
For a deep dive into the dataset metrics, bounding box cleaning logic, and exact class remapping rationales, please refer to the detailed report:
**`reports/preprocessing_report.md`**

## 9. Handover Checklist
Before beginning model training, verify the following are present and complete:
- [x] `data/CHV_yolo/data.yaml` exists.
- [x] `train`, `val`, and `test` images and labels exist in `data/CHV_yolo/`.
- [x] `reports/preprocessing_report.md` exists.
- [x] `requirements.txt` installed successfully.
