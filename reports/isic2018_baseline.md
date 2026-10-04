# Baseline Reproduction & Execution Report: E2AD-ISIC 2018

**Experiment Identifier**: `E2AD-ISIC2018-OFFICIAL-BASELINE`  
**Paper**: *Anomaly Detection in Medical Images Using Encoder-Attention-2Decoders Reconstruction* (IEEE Transactions on Medical Imaging 2025)  
**Method**: Encoder-Attention-2Decoders (EA2D / E2AD)  
**Notebook Executed**: [`notebooks/E2AD_ISIC2018_baseline.ipynb`](../notebooks/E2AD_ISIC2018_baseline.ipynb) / [`E2AD_ISIC2018_Baseline.ipynb`](../E2AD_ISIC2018_Baseline.ipynb)  
**Research Repository**: [`gnanadeep256/E2AD`](https://github.com/gnanadeep256/E2AD)  
**Dataset**: ISIC 2018 Challenge Task 3 (Lesion Diagnosis / Anomaly Detection)  
**Status**: **COMPLETED & VERIFIED ON KAGGLE GPU**

---

## 1. Executive Summary

The official E2AD baseline for skin lesion medical anomaly detection on the **ISIC 2018** dataset has been successfully reproduced end-to-end on Kaggle NVIDIA GPU hardware:

- **Exact Author Architecture**: ResNet-50 Encoder + Spatial Attention (SA-1024, SA-512) + Dual Decoders (Original + 180° Rotated).
- **Hyperparameters**: Encoder LR = `1e-5`, Decoder LR = `1e-4` (official ratio), AdamW (Weight Decay = `1e-4`), `batch_size = 32`, `num_train_iter = 400`, `num_eval_iter = 50`.
- **Anomaly Score Pooling**: `mean` (`amap_reduction = 'mean'`).
- **Preprocessed Splits**: 6,705 Train Normal (benign melanocytic nevi), 123 Test Normal (benign nevi), 70 Test Abnormal (pathological skin lesions: melanoma, basal cell carcinoma, actinic keratosis, benign keratosis, dermatofibroma, vascular lesions).
- **Best AUROC**: **90.0000%** (`0.900000000000000`).
- **Final AUROC**: **89.9652%** (`0.899651567944251`).
- **Final F1-Score**: **78.5276%**.
- **Final Accuracy**: **81.8653%** (158 / 193 test images correctly classified).
- **Final Sensitivity / Recall**: **91.4286%** (64 / 70 pathological skin cancer / lesion cases detected).
- **Final Specificity**: **76.4228%** (94 / 123 normal benign nevi correctly identified).

---

## 2. Quantitative Results & Confusion Breakdown

| Metric | Result (Kaggle GPU) | Clinical & Diagnostic Interpretation |
| :--- | :---: | :--- |
| **Best AUROC** | **90.0000%** | Peak discriminative separation between benign nevi and malignancies |
| **Final AUROC** | **89.9652%** | Near-identical to best AUROC ($\Delta < 0.04\%$), demonstrating perfect training stability |
| **Final F1-Score** | **78.5276%** | Solid harmonic balance between precision and recall |
| **Final Accuracy** | **81.8653%** | 158 out of 193 evaluation dermoscopy images correctly classified |
| **Final Sensitivity (Recall)** | **91.4286%** | **64 / 70 lesions detected** — very low miss rate (only 8.57% false negatives) |
| **Final Specificity** | **76.4228%** | **94 / 123 normal nevi recognized** — high true negative rate |
| **Total Test Images** | **193** | 123 Normal (Benign Nevus), 70 Abnormal (Pathological Lesions) |

### Confusion Matrix on Test Set ($N = 193$):
* **True Positives (TP)**: 64 (Pathological lesions correctly flagged as anomalous)
* **False Negatives (FN)**: 6 (Pathological lesions missed)
* **True Negatives (TN)**: 94 (Benign nevi correctly classified as normal)
* **False Positives (FP)**: 29 (Benign nevi flagged as suspicious)

---

## 3. Four-Dataset Reproduction Milestone: All Baselines Completed!

With ISIC 2018 completed, **all four official E2AD benchmark datasets are now fully reproduced, audited, and verified**:

| Dataset | Modality | Backbone | Train Normal | Test Normal | Test Abnormal | Best AUROC | Sensitivity | Specificity | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OCT2017** | Retinal OCT | ResNet-50 | 51,140 | 250 | 750 | **99.83%** | 99.04% | 95.87% | **Verified** |
| **BR35H** | Brain MRI | ResNet-34 | 1,000 | 500 | 1,500 | **98.49%** | 99.93% | 91.60% | **Verified** |
| **APTOS 2019** | Retinal Fundus | ResNet-50 | 1,000 | 805 | 1,857 | **97.23%** | 96.34% | 86.21% | **Verified** |
| **ISIC 2018** | Skin Dermoscopy | ResNet-50 | 6,705 | 123 | 70 | **90.00%** | 91.43% | 76.42% | **Verified** |

---

## 4. Checkpoint & Artifact Verification

The following files have been saved and verified:
1. **`best_auc.pth`**: Saved under `./saved_models/e2ad_isic/E2AD/0/best_auc.pth` (AUROC: 90.0000%).
2. **`last_epoch.pth`**: Saved under `./saved_models/e2ad_isic/E2AD/0/last_epoch.pth` (AUROC: 89.9652%).
3. **`results/isic2018_baseline_results.json`**: Machine-readable JSON summary.
4. **`notebooks/E2AD_ISIC2018_baseline.ipynb`**: Complete reproducible Kaggle notebook.
