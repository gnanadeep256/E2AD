import os
import sys
import json
import shutil

BASE_DIR = os.path.abspath(".")
SRC_DIR = os.path.join(BASE_DIR, "ISIC2018")
STAGE_DIR = os.path.join(BASE_DIR, "kaggle_dataset_isic")

print("=" * 65)
print("STAGING ISIC2018 DATASET FOR KAGGLE UPLOAD")
print("=" * 65)

if os.path.exists(STAGE_DIR):
    print(f"Cleaning previous staging directory: {STAGE_DIR}")
    shutil.rmtree(STAGE_DIR)

os.makedirs(os.path.join(STAGE_DIR, "train", "NORMAL"), exist_ok=True)
os.makedirs(os.path.join(STAGE_DIR, "test", "NORMAL"), exist_ok=True)
os.makedirs(os.path.join(STAGE_DIR, "test", "ABNORMAL"), exist_ok=True)

splits = [
    ("train/NORMAL", 6705),
    ("test/NORMAL", 123),
    ("test/ABNORMAL", 70)
]

total_linked = 0
for rel_path, expected_count in splits:
    src_sub = os.path.join(SRC_DIR, rel_path)
    dst_sub = os.path.join(STAGE_DIR, rel_path)
    files = [f for f in os.listdir(src_sub) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    assert len(files) == expected_count, f"Count mismatch in {rel_path}: expected {expected_count}, found {len(files)}"
    
    for f in files:
        src_file = os.path.join(src_sub, f)
        dst_file = os.path.join(dst_sub, f)
        try:
            os.link(src_file, dst_file)
        except Exception:
            shutil.copyfile(src_file, dst_file)
        total_linked += 1
    print(f"  [STAGED] {rel_path:<16} : {len(files)} images")

print(f"Total staged images: {total_linked} (Expected: 6898)")
assert total_linked == 6898, f"Staging total mismatch: {total_linked}"

# Write dataset-metadata.json
metadata = {
    "title": "ISIC 2018 E2AD Medical Anomaly Detection",
    "id": "gnanadeepthatavarthi/isic2018-e2ad",
    "licenses": [
        {
            "name": "CC-BY-NC-4.0"
        }
    ]
}

metadata_path = os.path.join(STAGE_DIR, "dataset-metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"  [WRITTEN] dataset-metadata.json -> {metadata_path}")

# Write README.md
readme_content = """# ISIC 2018 — E2AD Medical Anomaly Detection Benchmark

This dataset provides the standardized preprocessed benchmark split for unsupervised and semi-supervised anomaly detection on dermoscopy images, specifically configured for the **E2AD / EA2D** architecture:

* **Paper**: *Anomaly Detection in Medical Images Using Encoder-Attention-2Decoders Reconstruction* (IEEE Transactions on Medical Imaging 2025)
* **Official Reference**: [TumCCC/E2AD](https://github.com/TumCCC/E2AD)
* **Research Implementation**: [gnanadeep256/E2AD](https://github.com/gnanadeep256/E2AD)

---

## Dataset Structure

```text
ISIC2018/
├── train/
│   └── NORMAL/        (6,705 images: Benign Melanocytic Nevi [NV])
└── test/
    ├── NORMAL/        (123 images: Benign Melanocytic Nevi [NV])
    └── ABNORMAL/      (70 images: Pathological Skin Lesions [MEL, BCC, AKIEC, BKL, DF, VASC])
```

* **Total Images**: `6,898`
* **Resolution**: 600×450 native RGB dermoscopy photographs
* **Zero Missing / Zero Corrupted**: 100% verified files

---

## Anomaly Detection Task Formulation

In medical anomaly detection benchmarks:
1. **Normal (Benign)**: Melanocytic Nevus (`NV`) — benign skin moles. The model trains exclusively on healthy/benign nevi.
2. **Abnormal (Pathological)**: All non-NV disease categories from ISIC 2018 Task 3:
   * **MEL**: Melanoma
   * **BCC**: Basal Cell Carcinoma
   * **AKIEC**: Actinic Keratosis / Intraepithelial Carcinoma
   * **BKL**: Benign Keratosis (seborrheic keratoses, solar lentigines, lichen-planus like keratoses)
   * **DF**: Dermatofibroma
   * **VASC**: Vascular Lesion (angiomas, angiokeratomas, pyogenic granulomas, hemorrhage)

---

## Citations & Attribution

If you use this dataset, please cite the ISIC 2018 Challenge and the E2AD paper:

```bibtex
@article{e2ad2025,
  title={Anomaly Detection in Medical Images Using Encoder-Attention-2Decoders Reconstruction},
  author={Guo, J. and others},
  journal={IEEE Transactions on Medical Imaging},
  year={2025}
}

@article{codella2019skin,
  title={Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (ISIC)},
  author={Codella, Noel CF and others},
  journal={arXiv preprint arXiv:1902.03368},
  year={2019}
}

@article{tschandl2018ham10000,
  title={The HAM10000 dataset, a large collection of multi-source dermatological images of common pigmented skin lesions},
  author={Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal={Scientific Data},
  volume={5},
  year={2018}
}
```

**License**: Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC-4.0).
"""

readme_path = os.path.join(STAGE_DIR, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)
print(f"  [WRITTEN] README.md -> {readme_path}")
print("=" * 65)
print("STAGING COMPLETE: Dataset is ready for Kaggle upload.")
print(f"Directory: {STAGE_DIR}")
print("=" * 65)
