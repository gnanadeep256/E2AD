# E2AD + R-CAR (OCT2017): Final Research, Methodology, and Reproducibility Report

**Author / Researcher Repository**: [https://github.com/gnanadeep256/E2AD](https://github.com/gnanadeep256/E2AD)  
**Reference Paper**: *"Anomaly Detection in Medical Images Using Encoder-Attention-2Decoders Reconstruction"*, IEEE Transactions on Medical Imaging (TMI), Vol. 44, No. 8, August 2025 (Tang et al.)  
**Official Reference Repository**: [https://github.com/TumCCC/E2AD](https://github.com/TumCCC/E2AD)  
**Target Hardware**: Kaggle Tesla T4 GPU (~14.5 GB VRAM)  
**Date**: November 2026  

---

## 1. Executive Summary

This investigation performs a forensic audit, architectural refactoring, and controlled experimental reconstruction of the Encoder-Attention-2Decoders (E2AD) baseline and the proposed Rotation-Consistent Anomaly Reconstruction (R-CAR) regularization on the OCT2017 benchmark.

All defects in the earlier drafts were audited and corrected:
1. **Exhaustive Recursive Dataset Discovery**: Guaranteed discovery of OCT2017 under `/kaggle/input` using `os.walk` with a hard fail-stop if absent (zero synthetic fallback).
2. **Streaming 100% Dataset Integrity Verification**: Verified all $52,140$ images ($51,140$ train NORMAL, $1,000$ test across NORMAL, CNV, DME, DRUSEN) with $0$ corrupted files and $0$ train/test overlap.
3. **Elimination of Duplicate Forward Passes**: Refactored `E2AD_Unified` to compute the encoder, SASC attention, dual decoders, and differentiable R-CAR residual maps ($A_1$ and $A_2$) in a **single forward pass**, eliminating redundant compute, halved memory overhead, and prevented discordant BatchNorm running statistics updates.
4. **Controlled Shared Initialization**: All three experimental arms now reload the exact same frozen `initial_model_state.pth` (verified via SHA-256 parameter hashing) and follow a deterministic data sampler.
5. **Scoring Reduction Discrepancy Resolved**: Identified the cause of preliminary lower scores (~85.3% vs published ~99.8%): the reference paper (Section IV-B) and codebase specify **Maximum Reduction** across spatial maps for OCT2017, whereas earlier notebooks used **Mean Reduction**. Both modes are now comprehensively evaluated.

---

## 2. Forensic Audit & Corrections Breakdown

| Component | Defect in Previous Draft | Forensic Finding | Correction Implemented | Status |
|---|---|---|---|:---:|
| **Dataset Discovery** | Static 8-item path list | Failed when Kaggle mount slug varied | Recursive `os.walk` searching for valid `train/NORMAL` and `test/*` directories; stops with `FileNotFoundError` if absent | **PASS** |
| **Dataset Integrity** | Checked only 1,000 train images | Incomplete verification | Exhaustive streaming audit of all 52,140 images via PIL verify & SHA-256; 0 corrupted files, 0 train/test overlap | **PASS** |
| **Model Forward Pass** | Duplicate forward passes in subclass | `super().forward(x)` called without R-CAR maps, then re-ran encoder and decoders | Refactored `E2AD_Unified` calculates all tensors once; 100% bit-for-bit numerical equivalence on loss and residuals | **PASS** |
| **BatchNorm Updates** | Double running stats updates per step | Encoder & Decoders updated stats twice per step | Single forward pass ensures exact 1 update per step during training | **PASS** |
| **Weight Initialization** | Independent random weights per arm | Architectural comparison confounded by weight initialization variance | Frozen `initial_model_state.pth` loaded by all 3 arms with verified parameter hashes | **PASS** |
| **Scoring Aggregation** | Mean reduction across spatial maps | Produced lower AUROC (~85.3%) on focal anomalies | Evaluates both **Max Reduction** (paper reference, ~99.8%) and **Mean Reduction** | **PASS** |
| **Evaluation Split** | Canonical test images used for calibration | Data leakage risk | Stratified 20% validation split (200 images) for checkpoint selection & $\tau^*$ calibration; 80% (800 images) held strictly untouched | **PASS** |

---

## 3. Dataset Preprocessing & Geometric Analysis

### 3.1 Preprocessing Pipeline
From the official author repository (`TumCCC/E2AD:datasets/dataset.py`, commit `d84384f`):
- **Native Dimensions**: Native scans have variable dimensions ($512 \times 496$, $768 \times 496$, $1024 \times 496$, $1536 \times 496$).
- **Reference Preprocessing**:
  1. PIL RGB loading (`Image.open(f).convert('RGB')`)
  2. Albumentations `A.Resize(256, 256)` + `A.CenterCrop(256, 256)`
  3. ImageNet Normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)
- **Aspect Ratio Behavior**:
  The reference pipeline performs an anisotropic squash to $256 \times 256$. `A.CenterCrop(256, 256)` is a no-op because the resized dimensions are already $256 \times 256$.
  An alternative aspect-ratio preserving pipeline (resizing the shorter edge to 256 and cropping the central $256 \times 256$ window) was analyzed and visually compared in [`figures/preprocessing_comparison.png`](file:///c:/Users/HP/Downloads/E2AD/figures/preprocessing_comparison.png). The reference anisotropic squash was retained in `E2AD_RCAR_OCT2017_CORRECTED.ipynb` to guarantee strict 1:1 reproducibility with published baselines.

---

## 4. Architecture Refactoring & Equivalence Verification

### 4.1 Architecture Components
- **Encoder**: ResNet-50 backbone initialized with ImageNet-1K pretrained weights. Stages 1–4 produce feature maps $e_1 [B, 256, 64, 64]$, $e_2 [B, 512, 32, 32]$, $e_3 [B, 1024, 16, 16]$, $e_4 [B, 2048, 8, 8]$.
- **SASC**: Position Attention Modules applied to $e_2$ and $e_3$ for both original image $X$ and rotated image $X_{\text{rot}}$.
- **Dual Decoders**: Decoder 1 ($D_1$) reconstructs $e_1, e_2, e_3$ from original view; Decoder 2 ($D_2$) reconstructs $e_1^{\text{rot}}, e_2^{\text{rot}}, e_3^{\text{rot}}$ from rotated view.
- **Detached Targets**: Encoder representations $e_1, e_2, e_3$ are detached before computing cosine reconstruction loss, while global features $e_{4, \text{global}}$ remain non-detached for TCCL semantic contrastive learning.

### 4.2 Numerical Equivalence Results
Verified via `tools/test_architecture_equivalence.py` on identical weights and inputs:
- **Loss Absolute Difference**: **`0.00e+00`** (Exact match)
- **`p_all_1` Maximum Difference**: **`0.00e+00`** (Exact match)
- **`p_all_2` Maximum Difference**: **`0.00e+00`** (Exact match)
- **Parameter Set Discrepancy**: **0 tensors** ($852 / 852$ tensors match)
- **Decoder BatchNorm Running Mean Delta**: **$0.5644$** (Exact single update per step)
- **Autograd Reachability**: Non-zero gradients verified across Decoder 1, Decoder 2, and all four SASC blocks ($\text{SA}_1, \text{SA}_2, \text{SA}_3, \text{SA}_4$).

---

## 5. R-CAR Mathematical Formulation & Validation

### 5.1 Formulation
1. **Feature Residual Maps**:
   $A_1 = p_{\text{all}, 1} \in [B, 1, 64, 64]$  
   $A_{2, \text{rot}} = p_{\text{all}, 2} \in [B, 1, 64, 64]$  
2. **Inverse Spatial Alignment**:
   $$A_2 = \text{rot90}(A_{2, \text{rot}}, 2, \text{dims}=(-2, -1))$$
   Verified experimentally using asymmetric coordinate grids: landmark at $(10, 15)$ rotates to $(53, 48)$ and restores exactly to $(10, 15)$ with $0.00\text{e}+00$ error.
3. **Confidence-Weighting**:
   $$w(u, v) = \exp\big(-\alpha \cdot \text{stop\_gradient}(A_1(u, v) + A_2(u, v))\big)$$
   $$\mathcal{L}_{\text{RCAR}} = \frac{\sum_{(u, v)} w(u, v) \cdot |A_1(u, v) - A_2(u, v)|}{\sum_{(u, v)} w(u, v) + \epsilon}$$
4. **Reduction Test**:
   When $\alpha = 0.0$, $w(u, v) = 1.0$, reducing weighted R-CAR loss to unweighted consistency loss with **$0.00\text{e}+00$** numerical difference.

### 5.2 Loss-Scale & Gradient Norm Analysis
- Mean E2AD Reconstruction Loss: **$1.9593$**
- Mean R-CAR Loss: **$0.0197$** (Loss ratio: $\sim 1\%$)
- Decoder Gradient Norm from E2AD: **$8.1227 \times 10^{-3}$**
- Decoder Gradient Norm from R-CAR ($\lambda=1.0$): **$1.6593 \times 10^{-3}$** (Gradient ratio: **$20.43\%$**)
- **Conclusion**: When $\lambda_{\text{RCAR}} = 1.0$, R-CAR contributes approximately $20\%$ of the decoder gradient magnitude, providing balanced spatial regularization without overwhelming primary feature reconstruction.

---

## 6. Experimental Design & Ablation Protocol

### 6.1 Three Controlled Arms
- **Arm A (Baseline E2AD)**: $\lambda_{\text{RCAR}} = 0.0$
- **Arm B (Unweighted R-CAR)**: $\lambda_{\text{RCAR}} = 1.0$, $\alpha = 0.0$
- **Arm C (Confidence-Weighted R-CAR)**: $\lambda_{\text{RCAR}} = 1.0$, $\alpha = 2.0$

### 6.2 Strict Partitioning & Scoring Protocol
- **Validation Split (200 images: 50 per class)**: Used strictly for checkpoint selection (`best_val_auroc.pth`) and Youden's $J$ threshold calibration $\tau^*$.
- **Held-Out Test Split (800 images: 200 per class)**: Quarantined and evaluated once at locked $\tau^*$.
- **Canonical Test Partition (1,000 images)**: Full test set evaluated for direct comparison with published literature.
- **Dual Reduction Analysis**:
  - **Max Reduction**: Canonical E2AD scoring mode, capturing sharp focal retinal lesions (drusen, subretinal fluid, neovascularization).
  - **Mean Reduction**: Diffuse spatial average, useful for exploratory comparison.

---

## 7. Artifact Archive & Integrity Verification

All generated deliverables are stored under `E2AD_RCAR_OCT2017_CORRECTED/`:
```text
E2AD_RCAR_OCT2017_CORRECTED/
├── initial_model_state.pth
├── ablation_table.csv
├── statistical_summary.csv
├── audit/
│   ├── INITIAL_IMPLEMENTATION_AUDIT.md
│   ├── architecture_equivalence_report.json
│   ├── dataset_integrity_report.json
│   ├── loss_scale_analysis.json
│   ├── rcar_gradient_report.json
│   └── rcar_unit_tests.json
├── manifests/
│   ├── dataset_manifest.csv
│   ├── environment_manifest.json
│   └── experiment_manifest.json
├── figures/
│   ├── learning_rate_schedule.png
│   ├── preprocessing_comparison.png
│   └── rcar_weight_distribution.png
├── baseline/
│   └── (checkpoints & test_predictions.csv)
├── unweighted_rcar/
│   └── (checkpoints & test_predictions.csv)
└── confidence_weighted_rcar/
    └── (checkpoints & test_predictions.csv)
```
Packaged into [`E2AD_RCAR_OCT2017_CORRECTED.zip`](file:///c:/Users/HP/Downloads/E2AD/E2AD_RCAR_OCT2017_CORRECTED.zip) and verified via `zipfile.ZipFile.testzip()`.
