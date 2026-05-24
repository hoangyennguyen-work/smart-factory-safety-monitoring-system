# 🛡️ Smart Factory Safety Monitoring System

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/conda-enabled-green.svg)](https://conda.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ready-orange.svg)](https://github.com/ultralytics/ultralytics)

Welcome to the **Smart Factory Safety Monitoring System** repository. This system is designed to monitor factory work areas in real-time, focusing on detecting Personal Protective Equipment (PPE) like helmets and vests to ensure workplace safety.

Currently, this repository houses the **PPE Detection Module**, focusing on initial Exploratory Data Analysis (EDA), dataset cleaning, bounding-box validation, and preprocessing pipelines that prepare raw safety datasets for training state-of-the-art object detection models (YOLOv8).

---

## 📁 Repository Structure

*   `environment.yml` — Central Anaconda environment configuration file.
*   `requirements.txt` — Standard pip requirements file.
*   `ANACONDA_SETUP_GUIDE.md` — Complete, detailed step-by-step setup guide for Anaconda.
*   `ppe-detection/` — The primary PPE preprocessing module:
    *   `data/` — Dataset directories (raw & processed YOLO format).
    *   `notebooks/01_eda.ipynb` — Interactive Exploratory Data Analysis.
    *   `notebooks/02_preprocessing.ipynb` — Pipeline to clean, remap classes, and output YOLOv8 structured folders.
    *   `reports/preprocessing_report.md` — Comprehensive analysis of dataset distribution, remapping logic, and validation outcomes.

---

## 🚀 Quick Environment Setup (Anaconda)

We use **Anaconda** to handle complex native dependencies like OpenCV and Pillow easily on Windows. 

Open your **Anaconda Prompt** and execute these quick commands to set up the environment:

```cmd
# 1. Navigate to the project directory
cd C:\Github\smart-factory-safety-monitoring-system

# 2. Create the environment from configuration
conda env create -f environment.yml

# 3. Activate the environment
conda activate smart-factory-safety

# 4. Register the kernel for Jupyter/VS Code
python -m ipykernel install --user --name=smart-factory-safety --display-name "Python (smart-factory-safety)"
```

> [!TIP]
> For advanced configurations, manual setups, or troubleshooting, please see our dedicated [ANACONDA_SETUP_GUIDE.md](ANACONDA_SETUP_GUIDE.md).

---

## 📓 Running the Preprocessing Pipeline

Once the environment is active and the Jupyter kernel is registered, run the notebooks in order:

1.  **Exploratory Data Analysis**: Open `ppe-detection/notebooks/01_eda.ipynb` and run all cells to examine the dataset quality.
2.  **Dataset Preprocessing**: Open `ppe-detection/notebooks/02_preprocessing.ipynb` and run all cells to clean bounding boxes and compile the final YOLOv8 dataset.

The finalized, structured dataset will be generated automatically at:
📂 `ppe-detection/data/CHV_yolo/data.yaml`

---

## 🎯 Preprocessed Classes for YOLOv8
The original 6 color-specific helmet classes have been simplified into 3 core detection targets:
*   `0: person`
*   `1: helmet`
*   `2: vest`

*Note: Safety violations (e.g., detecting a person without a helmet) are handled dynamically using spatial intersection logic on the bounding boxes, eliminating the need for complex absence classes during model training.*