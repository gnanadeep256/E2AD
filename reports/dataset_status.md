# E2AD Dataset Status

## Environment

- **Python**: 3.11.15 (`.venv\Scripts\python.exe`)
- **NumPy**: 1.26.4
- **Pandas**: 3.0.6
- **OpenCV**: 4.11.0.86 (`opencv-python-headless`)
- **scikit-image**: 0.26.0
- **PyTorch**: Not installed (deferred to Kaggle GPU training environment)
- **Host Hardware**: Windows 11 | Intel Iris Xe Graphics | No NVIDIA CUDA GPU

---

## OCT2017

### Structure:
```
OCT2017/
├── train/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
└── test/
    ├── CNV/
    ├── DME/
    ├── DRUSEN/
    └── NORMAL/
```

### Counts:
- **Train Split**:
  - `NORMAL`: 51,140 images
  - `CNV`: 37,205 images
  - `DME`: 11,348 images
  - `DRUSEN`: 8,616 images
  - *Train Total*: **108,309** images
- **Test Split**:
  - `NORMAL`: 250 images
  - `CNV`: 250 images
  - `DME`: 250 images
  - `DRUSEN`: 250 images
  - *Test Total*: **1,000** images (250 Normal vs 750 Abnormal)
- **Dataset Total**: **109,309** `.jpeg` images
- **Sample Dimensions**:
  - Train: `(496, 512, 1)` (Grayscale mode `L`, loaded as RGB by `default_loader`)
  - Test: `(496, 1024, 1)` (Grayscale mode `L`, loaded as RGB by `default_loader`)
- **Corrupted / Zero-byte Images**: 0

### Status:
**Fully organized and verified.**
`AD_Dataset` natively loads `train/NORMAL` for unsupervised training and evaluates against all 4 classes in `test/` (treating `NORMAL` as label 0 and `CNV`, `DME`, `DRUSEN` as label 1). No preprocessing script is required.

---

## APTOS

### Raw structure:
```
APTOS/
└── original/
    ├── sample_submission.csv
    ├── test.csv
    ├── test_images/        (1,928 competition submission images, unused for AD)
    ├── train.csv           (3,662 rows: id_code, diagnosis)
    └── train_images/       (3,662 high-resolution fundus images)
```

### Counts:
- `train_images/`: 3,662 `.png` images
- Dimensions: High-resolution (e.g., `2136 x 3216 x 3`)
- `train.csv` Ground Truth Distribution:
  - Diagnosis `0` (Normal): 1,805 images
  - Diagnosis `1` (Mild DR): 370 images
  - Diagnosis `2` (Moderate DR): 999 images
  - Diagnosis `3` (Severe DR): 193 images
  - Diagnosis `4` (Proliferative DR): 295 images
  - Total Abnormal (1–4): 1,857 images
- Corrupted / Zero-byte Images: 0

### Required preprocessing:
- Script: `prepare_dataset/prepare_aptos.py`
- Algorithm:
  - Random seed fixed to `1` (`random.seed(1)`).
  - Normal images shuffled. First 1,000 assigned to `train/NORMAL`. Remaining 805 normal images assigned to `test/NORMAL`.
  - All 1,857 abnormal images assigned to `test/ABNORMAL`.
  - Every image transformed via `fundus_crop`: circular eye mask detection, bounding box extraction with 5px margin, and bilinear resize to `512 x 512`.

### Expected output:
```
APTOS/
├── train/
│   └── NORMAL/             (1,000 images, 512x512 PNG)
└── test/
    ├── NORMAL/             (805 images, 512x512 PNG)
    └── ABNORMAL/           (1,857 images, 512x512 PNG)
```
- **Total output images**: **3,662**

### Status:
**Raw data verified. Preprocessing pending.**

---

## BR35H

### Raw structure:
```
BR35H/
└── original/
    ├── no/                 (1,500 healthy brain MRI images)
    └── yes/                (1,500 brain tumor MRI images)
```

### Counts:
- `no/`: 1,500 `.jpg` images (sample shape: `630 x 630 x 3`)
- `yes/`: 1,500 `.jpg` images (sample shape: `348 x 287 x 3`)
- Total raw images: **3,000**
- Corrupted / Zero-byte Images: 0

### Required preprocessing:
- Script: `prepare_dataset/prepare_br35h.py`
- Algorithm:
  - Parses numeric ID from `no` filenames (`int("".join(filter(str.isdigit, file)))`).
  - IDs `< 1000` (1,000 images: `no0.jpg` to `no999.jpg`) $\rightarrow$ `train/NORMAL/`.
  - IDs $\ge 1000$ (500 images: `no1000.jpg` to `no1499.jpg`) $\rightarrow$ `test/NORMAL/`.
  - All 1,500 `yes` images $\rightarrow$ `test/ABNORMAL/`.
  - Operations: Direct lossless copy (`shutil.copyfile`), no image resize/crop.

### Expected output:
```
BR35H/
├── train/
│   └── NORMAL/             (1,000 images, original resolution JPG)
└── test/
    ├── NORMAL/             (500 images, original resolution JPG)
    └── ABNORMAL/           (1,500 images, original resolution JPG)
```
- **Total output images**: **3,000**

### Status:
**Fully preprocessed and verified.**
- Executed: `prepare_dataset/prepare_br35h.py --data-folder BR35H/original --save-folder BR35H`
- Verified Output:
  - `BR35H/train/NORMAL/`: 1,000 images (`no` images with numeric ID < 1000)
  - `BR35H/test/NORMAL/`: 500 images (`no` images with numeric ID >= 1000)
  - `BR35H/test/ABNORMAL/`: 1,500 images (all `yes` tumor images)
  - Total processed images: 3,000
  - Zero-byte / corrupted images: 0
  - Source bit-level match: 100% (0 size mismatches)
  - Raw `BR35H/original/` directory: Unchanged and fully preserved
- Status: **Ready for training.**

---

## ISIC2018

### Raw structure:
```
ISIC2018/
└── original/
    ├── ISIC2018_Task3_Training_GroundTruth/
    │   └── ISIC2018_Task3_Training_GroundTruth/
    │       ├── ATTRIBUTION.txt
    │       ├── ISIC2018_Task3_Training_GroundTruth.csv
    │       └── LICENSE.txt
    ├── ISIC2018_Task3_Training_Input/
    │   └── ISIC2018_Task3_Training_Input/
    │       ├── ATTRIBUTION.txt
    │       ├── LICENSE.txt
    │       └── 10,015 .jpg images
    ├── ISIC2018_Task3_Validation_GroundTruth/
    │   └── ISIC2018_Task3_Validation_GroundTruth/
    │       ├── ATTRIBUTION.txt
    │       ├── ISIC2018_Task3_Validation_GroundTruth.csv
    │       └── LICENSE.txt
    └── ISIC2018_Task3_Validation_Input/
        └── ISIC2018_Task3_Validation_Input/
            ├── ATTRIBUTION.txt
            ├── LICENSE.txt
            └── 193 .jpg images
```

### Counts:
- Training inputs: 10,015 `.jpg` images (sample shape: `450 x 600 x 3`)
- Validation inputs: 193 `.jpg` images
- Total images: **10,208**
- Ground Truth Breakdown (`line[2] == 1` is class `NV` = Melanocytic nevus / normal):
  - Training NV (Normal): 6,705 images
  - Training Non-NV (Malignant/Other lesions): 3,310 images (excluded from normal training)
  - Validation NV (Normal test): 123 images
  - Validation Non-NV (Abnormal test): 70 images
- Corrupted / Zero-byte Images: 0

### Required preprocessing:
- Script: `prepare_dataset/prepare_isic2018.py`
- Algorithm:
  - Reads `ISIC2018_Task3_Training_GroundTruth.csv`. Copies all `NV == 1` training images to `train/NORMAL/`.
  - Reads `ISIC2018_Task3_Validation_GroundTruth.csv`. Copies all `NV == 1` validation images to `test/NORMAL/` and all other classes to `test/ABNORMAL/`.
  - Operations: Direct lossless copy (`shutil.copyfile`).
- **Nesting Issue**: Extraction produced double-nested directories (`folder/folder/files`). The official script expects single-level paths (`source_dir/ISIC2018_Task3_Training_Input/*.jpg`).
- **Safe Normalization Procedure**: Either point the script to a flattened copy or safely promote the inner folders before execution.
- **Argparse Mismatch**: Script defines `--data_folder` and `--save_folder` (with underscores), while README uses hyphens (`--data-folder`).

### Expected output:
```
ISIC2018/
├── train/
│   └── NORMAL/             (6,705 images, original resolution JPG)
└── test/
    ├── NORMAL/             (123 images, original resolution JPG)
    └── ABNORMAL/           (70 images, original resolution JPG)
```
- **Total output images**: **6,898** (6,705 train + 193 test)

### Status:
**Raw data verified. Directory nesting normalization required prior to preprocessing.**

---

## Summary

| Dataset | Raw Status | Preprocessing Required | Expected Output | Current Status |
|---|---|---|---|---|
| **OCT2017** | 109,309 images organized in `train/` and `test/` | None (natively supported by `AD_Dataset`) | `train/` (108,309), `test/` (1,000) | **Ready for training** |
| **APTOS** | 3,662 images in `original/train_images` + `train.csv` | `prepare_aptos.py` (fundus crop & 512x512 resize) | `train/NORMAL` (1,000), `test/NORMAL` (805), `test/ABNORMAL` (1,857) | **Raw verified, preprocessing pending** |
| **BR35H** | 3,000 images in `original/no/` (1,500) and `original/yes/` (1,500) | Completed (`prepare_br35h.py`) | `train/NORMAL` (1,000), `test/NORMAL` (500), `test/ABNORMAL` (1,500) | **Fully processed & verified (ready for training)** |
| **ISIC2018** | 10,208 images across training/validation double-nested folders | Path normalization + `prepare_isic2018.py` | `train/NORMAL` (6,705), `test/NORMAL` (123), `test/ABNORMAL` (70) | **Raw verified, nesting normalization required** |
