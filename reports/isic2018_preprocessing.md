# ISIC2018 Dataset Preprocessing & Verification Report

**Dataset**: ISIC 2018 Challenge (Task 3: Skin Lesion Analysis Towards Melanoma Detection)  
**Execution Date**: 2026-10-03  
**Script**: [`prepare_dataset/prepare_isic2018.py`](../prepare_dataset/prepare_isic2018.py)  
**Environment**: Local Windows 11 / Python 3.11.15  
**Target Architecture**: E2AD (Encoder-Attention-2Decoders)

---

## 1. Executive Summary

The official ISIC2018 medical anomaly detection dataset has been preprocessed, structured, and audited locally. The raw Task 3 dataset contains dermoscopic photographs across 7 disease categories. In the unsupervised medical anomaly detection paradigm established by E2AD:
- **Normal Class**: Benign Melanocytic Nevi (`NV`).
- **Abnormal / Anomaly Class**: All other non-NV diagnostic categories (`MEL`, `BCC`, `AKIEC`, `BKL`, `DF`, `VASC`).

---

## 2. Source vs. Processed Counts

### Raw Source Dataset (Intact in `ISIC2018/original/`)
| Split / Component | Total Images | Class Composition | Integrity Status |
| :--- | :---: | :--- | :---: |
| **Training Input (`Task3_Training_Input`)** | **10,015** | 6,705 NV (Normal), 3,310 non-NV (Abnormal) | 0 corrupted, 100% verified |
| **Validation Input (`Task3_Validation_Input`)** | **193** | 123 NV (Normal), 70 non-NV (Abnormal) | 0 corrupted, 100% verified |
| **Total Raw Images** | **10,208** | Preserved completely intact | Verified unmodified |

### Processed Dataset for E2AD (`ISIC2018/`)
| Output Directory | Split Target | Role in E2AD | Exact Count | Verification |
| :--- | :--- | :--- | :---: | :---: |
| `train/NORMAL/` | Training Set | Unsupervised training (Normal only) | **6,705** | Match exact |
| `test/NORMAL/` | Test Set | Healthy evaluation controls (Label 0) | **123** | Match exact |
| `test/ABNORMAL/` | Test Set | Pathological anomaly evaluation (Label 1) | **70** | Match exact |
| **Total Preprocessed** | — | **Full E2AD Dataset** | **6,898** | **100% Match** |

---

## 3. Directory Layout

```
ISIC2018/
├── original/                         # Raw downloaded files (preserved intact)
│   ├── ISIC2018_Task3_Training_GroundTruth/
│   │   └── ISIC2018_Task3_Training_GroundTruth/
│   │       └── ISIC2018_Task3_Training_GroundTruth.csv
│   ├── ISIC2018_Task3_Training_Input/
│   │   └── ISIC2018_Task3_Training_Input/
│   │       └── [10,015 .jpg images]
│   ├── ISIC2018_Task3_Validation_GroundTruth/
│   │   └── ISIC2018_Task3_Validation_GroundTruth/
│   │       └── ISIC2018_Task3_Validation_GroundTruth.csv
│   └── ISIC2018_Task3_Validation_Input/
│       └── ISIC2018_Task3_Validation_Input/
│           └── [193 .jpg images]
├── train/
│   └── NORMAL/                       # 6,705 benign melanocytic nevus images
└── test/
    ├── NORMAL/                       # 123 benign melanocytic nevus images
    └── ABNORMAL/                     # 70 pathological skin lesion images
```

---

## 4. Automated Integrity Audit Results

1. **Zero-Byte Files**: Exactly **0** zero-byte files found across all 6,898 preprocessed images.
2. **Corrupted / Unreadable Files**: Exactly **0** unreadable files.
3. **Color Channels**: 100% of images are 3-channel RGB (`Image.mode == 'RGB'`).
4. **Resolution**: Native resolution is $600 \times 450$ pixels.
5. **No Data Leakage**: Training normal and validation normal sets are derived from mutually exclusive official ISIC challenge splits.
6. **Raw Data Preservation**: Verified that `ISIC2018_Task3_Training_Input` contains exactly 10,015 images and `ISIC2018_Task3_Validation_Input` contains exactly 193 images, confirming raw data was not modified.
