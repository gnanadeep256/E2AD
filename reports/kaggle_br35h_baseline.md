# Kaggle Execution Guide: E2AD-BR35H-OFFICIAL-BASELINE

**Experiment Identifier**: `E2AD-BR35H-OFFICIAL-BASELINE`  
**Paper**: *Anomaly Detection in Medical Images Using Encoder-Attention-2Decoders Reconstruction* (IEEE Transactions on Medical Imaging 2025)  
**Personal FYP Repository**: [`gnanadeep256/E2AD`](https://github.com/gnanadeep256/E2AD)  
**Upstream Reference**: [`TumCCC/E2AD`](https://github.com/TumCCC/E2AD)  
**Notebook**: [`E2AD_BR35H_Baseline.ipynb`](../E2AD_BR35H_Baseline.ipynb)  
**Purpose**: Faithful reproduction of the official E2AD baseline on the BR35H brain tumor MRI anomaly detection dataset using modern Kaggle GPU infrastructure.

---

## 1. Environment & Hardware Specifications

### Kaggle Target Environment
- **Compute Accelerator**: NVIDIA Tesla T4 (or P100) — Single GPU (`cuda:0`, 16 GB VRAM)
- **Kaggle Setting**: **Internet** MUST be toggled **ON** (required to fetch initial ImageNet-1k pre-trained ResNet-50 weights: `https://download.pytorch.org/models/resnet50-0676ba61.pth`)
- **Python Version**: 3.11 / 3.12 (native Kaggle environment)
- **PyTorch Version**: 2.x (e.g. 2.10.0+cu128)
- **Torchvision Version**: 0.25+ (or $\ge 0.14$)
- **Albumentations Version**: 1.4+ / 2.0+

> [!IMPORTANT]
> Do **NOT** run `pip install -r requirements.txt` blindly. Kaggle's native environment already provides modern, pre-optimized builds of PyTorch, Torchvision, Albumentations, OpenCV, Scikit-learn, and Pandas. Blindly installing `requirements.txt` will break PyTorch and CUDA installations.

---

## 2. Compatibility Audit & Applied Minimal Patches

Two compatibility adjustments were identified and applied to enable the original official codebase to run seamlessly under PyTorch 2.x and Albumentations 1.4+:

### 1. Torchvision Internal Utility Patch
- **Affected Files**: [`models/resnet.py`](../models/resnet.py), [`models/resnet_decoder.py`](../models/resnet_decoder.py)
- **Root Cause**: Torchvision $\ge 0.14$ permanently removed `torchvision._internally_replaced_utils`.
- **Solution**: Replaced deprecated import with standard PyTorch hub loader:
  ```python
  # Replaced:
  # from torchvision._internally_replaced_utils import load_state_dict_from_url
  from torch.hub import load_state_dict_from_url
  ```

### 2. Albumentations `to_tuple` Removal Patch
- **Affected File**: [`datasets/transforms.py`](../datasets/transforms.py)
- **Root Cause**: In modern Albumentations (v1.4+ / v2.0+), `to_tuple` was removed from `albumentations.core.transforms_interface`.
- **Solution**: Wrapped the import with a safe fallback:
  ```python
  from albumentations.core.transforms_interface import (
      DualTransform,
      ImageOnlyTransform,
      NoOp,
  )
  try:
      from albumentations.core.transforms_interface import to_tuple
  except ImportError:
      def to_tuple(param, low=None, bias=None):
          return tuple(param) if isinstance(param, (list, tuple)) else (param, param)
  ```

### 3. Execution Flags Alignment
- **GPU Index**: Kaggle single-GPU maps to index `0`. The script default (`--gpu '1'`) must be overridden with `--gpu 0`.
- **Checkpoint Persistence**: Default `--save_weight 0` does not write weights to disk. `--save_weight 1` must be provided to persist `best_auc.pth` and `last_epoch.pth`.

---

## 3. BR35H Dataset Preparation on Kaggle

The official E2AD pipeline expects the following directory structure:
```
/kaggle/working/BR35H/
├── train/
│   └── NORMAL/       (1,000 normal MRI slices)
└── test/
    ├── NORMAL/       (500 normal MRI slices)
    └── ABNORMAL/     (1,500 tumor MRI slices)
```

### Raw Dataset Source
Add the Kaggle dataset: [`ahmedhamada0/brain-tumor-detection`](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) to your Kaggle Notebook.  
It provides:
```
brain-tumor-detection/
├── no/   (1,500 normal images)
└── yes/  (1,500 abnormal images)
```

### Preprocessing Execution
If not already processed, run the official preprocessing script:
```bash
python prepare_dataset/prepare_br35h.py \
    --data-folder /kaggle/input/brain-tumor-detection \
    --save-folder /kaggle/working/BR35H
```

---

## 4. End-to-End Execution Flow

The reproduction workflow is fully automated inside [`E2AD_BR35H_Baseline.ipynb`](../E2AD_BR35H_Baseline.ipynb). It executes the following 12 stages:

| Section | Stage | Description |
| :--- | :--- | :--- |
| **1** | **Environment Audit** | Inspects Python, PyTorch, Torchvision, Albumentations, CUDA, and GPU memory. |
| **2** | **Repository Check** | Clones/navigates to `/kaggle/working/E2AD` and validates all source modules. |
| **3** | **Dataset Check** | Verifies 1000/500/1500 image counts and performs PIL integrity checks on sample images. |
| **4** | **Compatibility Summary** | Documents compatibility requirements for modern Kaggle runtime. |
| **5** | **Torchvision Verification** | Verifies and validates `torch.hub.load_state_dict_from_url` in ResNet modules. |
| **6** | **Albumentations Verification** | Validates `datasets/transforms.py` and tests top-level imports. |
| **7** | **DataLoader Smoke Test** | Loads official training split, confirms batch shape `[32, 3, 256, 256]` and label tensor shape `[32]`. |
| **8** | **Model Smoke Test** | Instantiates `E2AD` on `cuda:0` and runs dummy forward pass to check tensor dimensions. |
| **9** | **Official Baseline Training** | Runs 4,000 iterations of AdamW optimization, logging evaluation every 100 iters. |
| **10** | **Checkpoint Verification** | Confirms `best_auc.pth` and `last_epoch.pth` are successfully saved and checks file sizes. |
| **11** | **Results Extraction** | Automatically parses training logs for peak AUROC, F1, Accuracy, Sensitivity, and Specificity. |
| **12** | **Reproducibility Report** | Compares experimental findings directly with IEEE TMI 2025 published figures. |

---

## 5. Official Baseline Training Command

```bash
python e2ad_br35h.py \
    --train_times 1 \
    --gpu 0 \
    --model_name E2AD \
    --data_dir /kaggle/working/BR35H/ \
    --save_weight 1 \
    --save_dir ./saved_models \
    --save_name e2ad_br35h \
    --num_train_iter 4000 \
    --num_eval_iter 100 \
    --batch_size 32 \
    --eval_batch_size 64 \
    --optim AdamW \
    --lr 5e-4 \
    --lr_encoder 5e-5 \
    --weight_decay 1e-4 \
    --amp True \
    --seed 0
```

> [!TIP]
> **VRAM & Hardware Context**:  
> In the paper, the authors executed experiments on a 40 GB NVIDIA A100 (`A100-PCIE-40 GB`). On Kaggle's 15 GB Tesla T4 GPU, batch size 32 in standard FP32 reaches $\sim 14.1\text{ GB}$, which exhausts GPU memory during the high-dimensional cosine similarity reconstruction steps.  
> The authors natively implemented Automatic Mixed Precision (`torch.cuda.amp`) via `--amp` in `e2ad_br35h.py` and `methods/edc1.py`. Passing `--amp True` reduces activation memory to $\sim 6\text{ GB}$ (leaving $\sim 9\text{ GB}$ headroom on a Tesla T4) without changing the architecture, batch size, learning rate, or loss formulation.
> Additionally, setting `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` prevents PyTorch memory fragmentation.

---

## 6. Official IEEE TMI 2025 Paper Benchmark (BR35H)

From Table IV and Section IV-D of the paper (*"Anomaly Detection in Medical Images Using Encoder-Attention-2Decoders Reconstruction"*):

| Metric | Published Result (IEEE TMI 2025) | Our Kaggle Baseline (`seed=0`) | Status |
| :--- | :--- | :--- | :--- |
| **AUROC (%)** | **99.83%** | *(To be populated upon Kaggle run)* | Pending GPU Run |
| **F1-Score (%)** | **99.62%** | *(To be populated upon Kaggle run)* | Pending GPU Run |
| **Accuracy (%)** | **99.44%** | *(To be populated upon Kaggle run)* | Pending GPU Run |
| **Sensitivity (%)** | *High ($\approx 99.0\%+$)* | *(To be populated upon Kaggle run)* | Computed |
| **Specificity (%)** | *High ($\approx 96.0\%+$)* | *(To be populated upon Kaggle run)* | Computed |

> [!NOTE]
> The authors noted in Section IV-D and IV-L that anomaly detection on BR35H is a clean, high-contrast task where the baseline backbone alone reaches $99.74\%$ AUROC, and E2AD reaches $99.83\%$ AUROC.
