# Environment & Dependency Audit: E2AD Reproduction

**Date**: September 29, 2026  
**Project**: Anomaly Detection in Medical Images Using Encoder-Attention-2Decoders Reconstruction (E2AD / EA2D)  
**Host Machine**: Windows 11 | Global Python 3.14.0 | Python 3.11.15 Available | Intel Iris Xe Graphics (No CUDA GPU)  
**Target Training Platform**: Kaggle GPU (NVIDIA T4 / P100)  

---

## 1. Project-Local Virtual Environment Verification

The local virtual environment `.venv` has been successfully provisioned using Python 3.11 and verified:

- **Virtual Environment Tool**: `uv` (`uv venv --seed --clear --python 3.11 .venv`)
- **Python Version**: `Python 3.11.15`
- **Interpreter Path**: `C:\Users\HP\Downloads\E2AD\.venv\Scripts\python.exe`
- **Pip Version**: `pip 26.2.1 from C:\Users\HP\Downloads\E2AD\.venv\Lib\site-packages\pip (python 3.11)`

---

## 2. Comprehensive Dependency Audit

Below is the complete analysis of all dependencies, comparing the author's original `requirements.txt` against code imports, Python 3.11 compatibility, Windows compatibility, and Kaggle environment defaults.

| Dependency | Original `requirements.txt` | Imported By | Python 3.11 Compatibility | Windows Compatibility | Kaggle Compatibility | Audit Assessment & Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`torch`** | `1.12.0+cu113` | `models/`, `methods/`, `datasets/`, `train_utils.py`, `utils.py`, `e2ad_*.py` | ❌ **Incompatible** (PyTorch 1.12 wheels stop at Python 3.10) | ❌ Incompatible for Py 3.11 | ⚠️ Kaggle default has PyTorch 2.1+ | **Do not install old version.** For local code inspection / dry-runs, install PyTorch CPU (`torch>=2.0.0`). On Kaggle, use Kaggle's native PyTorch GPU runtime. |
| **`torchvision`** | `0.13.0+cu113` | `models/resnet.py`, `models/resnet_decoder.py`, `datasets/`, `utils.py` | ❌ **Incompatible** (No Py 3.11 wheel for 0.13.0) | ❌ Incompatible for Py 3.11 | ⚠️ Kaggle default has `torchvision>=0.16` | **Critical code issue:** In `torchvision>=0.14`, `from torchvision._internally_replaced_utils import load_state_dict_from_url` was deleted. Must use `from torch.hub import load_state_dict_from_url`. |
| **`albumentations`** | ⚠️ **MISSING** | `datasets/dataset.py`, `datasets/transforms.py` | ✅ Compatible (`>=1.3.0`) | ✅ Compatible | ✅ Pre-installed on Kaggle | **Critical repo omission.** The code crashes without it. Must be explicitly installed locally and in Kaggle environments. |
| **`pyyaml`** | ⚠️ **MISSING** | `utils.py` (`import yaml`) | ✅ Compatible (`>=6.0`) | ✅ Compatible | ✅ Pre-installed on Kaggle | **Critical repo omission.** Required by `over_write_args_from_file`. Must be explicitly installed. |
| **`numpy`** | `1.18.4` | Used throughout all modules | ❌ **Incompatible** (numpy 1.18 only supports up to Python 3.8) | ❌ Cannot compile on Windows Py 3.11 | ⚠️ Kaggle uses `numpy>=1.24` | Install modern `numpy>=1.23.2,<2.0.0`. Pin `<2.0.0` to avoid ABI breaking changes with older compiled libraries. |
| **`pandas`** | `1.3.5` | `prepare_dataset/`, `datasets/dataset.py` | ❌ **Incompatible** (pandas 1.3 only supports up to Python 3.10) | ❌ Cannot install on Py 3.11 | ✅ Kaggle uses `pandas>=2.0` | Install modern `pandas>=1.5.0`. |
| **`opencv-python-headless`** | `4.6.0.66` | `prepare_dataset/`, `methods/edc1.py`, `datasets/dataset.py` | ✅ Compatible (modern `>=4.8.0`) | ✅ Compatible | ✅ Pre-installed on Kaggle | Install `opencv-python-headless>=4.8.0`. |
| **`scikit-image`** | `0.19.3` | `prepare_dataset/prepare_aptos.py` | ✅ Compatible (`>=0.20.0`) | ✅ Compatible | ✅ Pre-installed on Kaggle | Install `scikit-image>=0.20.0` (required for `regionprops`, `label`). |
| **`scikit-learn`** | `0.22.2.post1` | `methods/edc1.py` (`from sklearn.metrics import *`) | ❌ **Incompatible** (0.22 from 2020 cannot build on Py 3.11) | ❌ Cannot compile | ✅ Pre-installed on Kaggle (`>=1.2`) | Install modern `scikit-learn>=1.2.0` (API for `roc_auc_score`, `precision_recall_curve`, `f1_score` is backward-compatible). |
| **`Pillow`** | `9.0.1` | `datasets/dataset.py` | ❌ **Incompatible** (Pillow 9.0 has no Py 3.11 wheels) | ❌ Fails on Py 3.11 | ✅ Pre-installed on Kaggle | Install `Pillow>=9.3.0`. |
| **`matplotlib`** | `3.2.1` | `custom_writer.py`, `methods/edc1.py`, `datasets/dataset.py` | ❌ **Incompatible** (3.2 stops at Py 3.8) | ❌ Cannot compile | ✅ Pre-installed on Kaggle | Install `matplotlib>=3.6.0`. |
| **`scipy`** | `1.4.1` | Transitive dependency for scikit-learn / scikit-image | ❌ **Incompatible** (1.4 stops at Py 3.8) | ❌ Cannot compile | ✅ Pre-installed on Kaggle | Install `scipy>=1.9.2`. |
| **`tensorboard`** | `2.11.0` | `train_utils.py`, `utils.py` | ✅ Compatible (`>=2.11.0`) | ✅ Compatible | ✅ Pre-installed on Kaggle | Required for `torch.utils.tensorboard.SummaryWriter`. |
| **`tqdm`** | `4.64.1` | Dataset and loop utilities | ✅ Compatible (`>=4.64.1`) | ✅ Compatible | ✅ Pre-installed on Kaggle | Install `tqdm>=4.64.1`. |
| **`tabulate`** | `0.9.0` | Utility formatting | ✅ Compatible (`>=0.9.0`) | ✅ Compatible | ✅ Compatible | Install `tabulate>=0.9.0`. |
| **`ptflops`** | `0.7` | FLOPs profiling | ✅ Compatible (`>=0.7`) | ✅ Compatible | ✅ Compatible | Optional profiling package. |

---

## 3. Two-Tier Environment Plans

### Plan 1: Local Development Environment (Windows 11)
**Purpose**:
- Preprocess datasets (`APTOS`, `Br35H`, `ISIC2018`).
- Verify dataset directories and sample integrity.
- Inspect and static-check model code without CUDA.
- Run lightweight CPU forward-pass smoke tests if needed.

**Configuration**:
- **Python**: `3.11.15` in `.venv`
- **Phase 1 Packages (Dataset Preprocessing ONLY - No PyTorch required yet)**:
  - `numpy<2.0.0`
  - `pandas>=1.5.0`
  - `opencv-python-headless>=4.8.0`
  - `scikit-image>=0.20.0`
- **Phase 2 Packages (Optional Local Code Inspection / Smoke Testing)**:
  - `torch>=2.0.0` (CPU-only wheel from PyTorch official index)
  - `torchvision>=0.15.0` (CPU-only wheel)
  - `albumentations>=1.3.0`
  - `Pillow>=9.3.0`
  - `scikit-learn>=1.2.0`
  - `pyyaml>=6.0`
  - `matplotlib>=3.6.0`
  - `tensorboard>=2.11.0`

---

### Plan 2: Kaggle Training Environment (GPU Platform)
**Purpose**:
- Full model training with GPU acceleration.
- Multi-run experiments (`--train_times 5`).
- Official baseline evaluation and checkpoint generation.

**Configuration**:
- **Environment**: Kaggle Notebook / Script with GPU accelerator (NVIDIA Tesla T4 $\times 2$ or P100 16GB)
- **Python**: Python 3.10 / 3.11 (Kaggle default standard)
- **PyTorch / CUDA**: Pre-installed PyTorch 2.1+ / 2.4+ with CUDA 12.1 / 11.8
- **Packages**:
  - Kaggle pre-installs `torch`, `torchvision`, `numpy`, `pandas`, `opencv-python`, `scikit-learn`, `matplotlib`, `albumentations`, `pyyaml`.
- **Known Code Compatibility Risks on Kaggle & Solutions**:
  1. **Torchvision import**: Replace `from torchvision._internally_replaced_utils import load_state_dict_from_url` with `from torch.hub import load_state_dict_from_url` in `models/resnet.py` and `models/resnet_decoder.py`.
  2. **GPU ID default**: The scripts default to `--gpu 1`. On single-GPU Kaggle sessions, `--gpu 0` must be explicitly passed.
  3. **Timing Events**: `torch.cuda.Event` and `torch.cuda.synchronize()` in `methods/edc1.py` require a CUDA-enabled PyTorch build (fully supported on Kaggle GPU).
  4. **Dataset Upload**: Preprocessed datasets (`APTOS`, `Br35H`, `ISIC2018`, `OCT2017`) can be zipped locally and uploaded as a Kaggle private dataset, avoiding redundant preprocessing overhead on the GPU instance.

---

## 4. Current Status & Safety Guarantees

- [x] No source code was modified.
- [x] No dataset files were altered or moved.
- [x] `requirements.txt` remains strictly unchanged.
- [x] Old incompatible packages were **not** installed.
- [x] Clean `.venv` with Python 3.11 is verified and ready.
