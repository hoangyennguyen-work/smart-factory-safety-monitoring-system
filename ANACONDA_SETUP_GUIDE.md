# Anaconda Setup Guide — Smart Factory Safety Monitoring System

This guide outlines how to set up the **smart-factory-safety** Anaconda environment to run the Exploratory Data Analysis (EDA) and preprocessing notebooks in this repository.

By using Anaconda, we take advantage of pre-compiled, optimized binaries for performance-critical libraries like `numpy` and `opencv` on Windows, which helps avoid common build issues.

---

## 🛠️ Prerequisites
Make sure you have either **Anaconda** or **Miniconda** installed on your system.
* You can verify this by opening your search bar, typing **Anaconda Prompt**, and opening it.
* Alternatively, run `conda --version` in your terminal.

---

## 🚀 Step-by-Step Setup

### Step 1: Open Anaconda Prompt
Open the **Anaconda Prompt** (or **Miniconda Prompt**) from your Start Menu. This ensures all conda commands are loaded in your shell path.

### Step 2: Navigate to the Repository Root
Change your directory to where the project is located:
```cmd
cd C:\Github\smart-factory-safety-monitoring-system
```

### Step 3: Create the Conda Environment
Use the provided `environment.yml` file to create a clean, dedicated environment:
```cmd
conda env create -f environment.yml
```
> [!NOTE]
> This command will download and configure Python 3.10 along with the standard ML & Computer Vision dependencies (Pandas, Seaborn, Matplotlib, Pillow, OpenCV, Jupyter, and IPykernel) from the `conda-forge` channel.

### Step 4: Activate the Environment
Once the environment is created, activate it:
```cmd
conda activate smart-factory-safety
```

### Step 5: Register the Jupyter Kernel (Crucial Step)
To ensure that Jupyter Notebook or VS Code can locate this specific environment, register it as a Jupyter kernel:
```cmd
python -m ipykernel install --user --name=smart-factory-safety --display-name "Python (smart-factory-safety)"
```

---

## 🎮 Enabling NVIDIA GPU Acceleration (CUDA)
By default, the environment installs a CPU-only version of PyTorch. To train your YOLO models at maximum speed on an NVIDIA GPU (e.g. RTX 30, 40, or 50 series), you need to install PyTorch with CUDA support.

### Standard GPUs (RTX 30 / 40 Series — CUDA 12.1 / 12.4)
Run this command to install PyTorch with CUDA 12.1 support:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### Next-Gen GPUs (RTX 50 Series Blackwell — CUDA 13.2)
If you are using an NVIDIA RTX 50-series Blackwell GPU (like the RTX 5060 Ti) and want to use the experimental CUDA 13.2 runtime, run the PyTorch nightly build command:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132 --force-reinstall
```

### 🔍 Verification Check
To verify that PyTorch is successfully recognizing your NVIDIA GPU:
```cmd
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('Supported Architectures:', torch.cuda.get_arch_list())"
```
For RTX 50-series Blackwell cards, make sure `sm_120` appears in the list of supported architectures.

---

## 📓 Running the Notebooks

### Option A: Using VS Code (Recommended)
1. Open the repository root folder in VS Code.
2. Open the first notebook: `ppe-detection/notebooks/01_eda.ipynb`.
3. In the top right corner of the notebook editor, click **Select Kernel** -> **Python Environments...**
4. Choose **Python (smart-factory-safety)** (or the interpreter pointing to the `smart-factory-safety` conda environment).
5. Run the cells.

### Option B: Using Jupyter Notebook Classic
1. In your activated Anaconda prompt, run:
   ```cmd
   jupyter notebook
   ```
2. In the web interface, open `ppe-detection/notebooks/01_eda.ipynb` or `02_preprocessing.ipynb`.
3. If the kernel is not set automatically, go to the top menu: **Kernel** -> **Change kernel** -> **Python (smart-factory-safety)**.

---

## ⚡ Alternative Quick Setup (Command Line)
If you prefer not to use the `environment.yml` file, you can create the environment manually via CLI:

```cmd
# Create the environment with Python 3.10
conda create -y -n smart-factory-safety python=3.10

# Activate the environment
conda activate smart-factory-safety

# Install the dependencies from requirements.txt
pip install -r requirements.txt

# Register the Jupyter kernel
python -m ipykernel install --user --name=smart-factory-safety --display-name "Python (smart-factory-safety)"
```

---

## 🔍 Validation Check
To verify that everything is working perfectly, you can run a quick check within your activated environment:
```cmd
python -c "import cv2, pandas, PIL, matplotlib, seaborn; print('All libraries loaded successfully!')"
```
This should print `All libraries loaded successfully!` without any errors. You are now fully prepared to run the safety monitoring preprocessing pipeline!
