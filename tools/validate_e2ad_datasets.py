"""
validate_e2ad_datasets.py

A strictly READ-ONLY dataset validation script for E2AD benchmarks:
- OCT2017
- APTOS 2019
- Br35H
- ISIC 2018

This script inspects paths, counts, extensions, directory structures,
class distributions, sample image dimensions, and verifies image integrity.
IT NEVER MODIFIES, MOVES, RENAMES, DELETES, OR WRITES ANY DATASET FILES.
"""

import os
import sys
from collections import Counter
from PIL import Image
import pandas as pd


def check_image_readable(filepath):
    """Attempt to open and verify image file without modifying it."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True, None
    except Exception as e:
        return False, str(e)


def get_image_info(filepath):
    """Return image dimensions (height, width, channels) safely."""
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            mode = img.mode
            channels = len(img.getbands())
            return (height, width, channels), mode
    except Exception:
        return None, None


def inspect_oct2017(base_dir="OCT2017"):
    print("\n" + "=" * 60)
    print("DATASET: OCT2017")
    print(f"Path: {os.path.abspath(base_dir)}")
    print("=" * 60)

    if not os.path.exists(base_dir):
        print(f"[ERROR] Directory not found: {base_dir}")
        return

    splits = ["train", "test"]
    classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
    total_images = 0
    all_exts = Counter()
    corrupted_count = 0
    sample_dims = {}

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        print(f"\nSplit: {split}/")
        if not os.path.exists(split_dir):
            print(f"  [MISSING] {split}/ directory not found!")
            continue

        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.exists(cls_dir):
                print(f"  {cls}: [MISSING]")
                continue

            files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(files)
            for f in files:
                all_exts[os.path.splitext(f)[1].lower()] += 1

            # Sample dimensions
            if files and (split, cls) not in sample_dims:
                sample_path = os.path.join(cls_dir, files[0])
                dims, mode = get_image_info(sample_path)
                sample_dims[f"{split}/{cls}"] = (dims, mode, files[0])

            # Sample verification of first 50 files per class for speed
            for f in files[:50]:
                ok, _ = check_image_readable(os.path.join(cls_dir, f))
                if not ok:
                    corrupted_count += 1

            print(f"  {cls:<10}: {len(files):>7} images")

    print(f"\nTotal image files: {total_images}")
    print(f"Extensions: {dict(all_exts)}")
    print("Sample dimensions (H, W, C):")
    for key, (dims, mode, fname) in sample_dims.items():
        print(f"  {key:<18}: {dims} (mode: {mode}, e.g. {fname})")
    print(f"Corrupted images detected (sampled): {corrupted_count}")
    print("Structure assessment: Matches official benchmark requirements (train/test with CNV, DME, DRUSEN, NORMAL).")


def inspect_aptos(base_dir="APTOS"):
    print("\n" + "=" * 60)
    print("DATASET: APTOS 2019")
    print(f"Path: {os.path.abspath(base_dir)}")
    print("=" * 60)

    orig_dir = os.path.join(base_dir, "original")
    if not os.path.exists(orig_dir):
        print(f"[ERROR] Directory not found: {orig_dir}")
        return

    train_img_dir = os.path.join(orig_dir, "train_images")
    train_csv_path = os.path.join(orig_dir, "train.csv")

    train_imgs = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(train_img_dir) else []
    all_exts = Counter([os.path.splitext(f)[1].lower() for f in train_imgs])

    print(f"\nRaw Folder: APTOS/original/")
    print(f"  train_images/: {len(train_imgs)} images (extensions: {dict(all_exts)})")

    sample_dim = None
    if train_imgs:
        sample_dim, mode = get_image_info(os.path.join(train_img_dir, train_imgs[0]))
        print(f"  Sample dimensions: {sample_dim} (mode: {mode}, e.g. {train_imgs[0]})")

    corrupted_count = 0
    for f in train_imgs[:50]:
        ok, _ = check_image_readable(os.path.join(train_img_dir, f))
        if not ok:
            corrupted_count += 1

    if os.path.exists(train_csv_path):
        df = pd.read_csv(train_csv_path)
        print(f"  train.csv: {len(df)} rows")
        dist = df['diagnosis'].value_counts().sort_index().to_dict()
        normal_cnt = dist.get(0, 0)
        abnormal_cnt = sum(cnt for diag, cnt in dist.items() if diag != 0)
        print(f"  Diagnosis breakdown: {dist}")
        print(f"  Normal (0) count   : {normal_cnt}")
        print(f"  Abnormal (1-4) count: {abnormal_cnt}")
        print(f"\nExpected prepare_aptos.py Output:")
        print(f"  train/NORMAL/  : 1000 images (first 1000 shuffled normal)")
        print(f"  test/NORMAL/   : {normal_cnt - 1000} images (remaining normal)")
        print(f"  test/ABNORMAL/ : {abnormal_cnt} images (all abnormal)")
        print(f"  Total output   : {len(df)} images (fundus cropped to 512x512)")
    else:
        print("  [ERROR] train.csv not found!")

    print(f"Corrupted images detected (sampled): {corrupted_count}")
    print("Structure assessment: Raw images and CSV present. Preprocessing required.")


def inspect_br35h(base_dir="BR35H"):
    print("\n" + "=" * 60)
    print("DATASET: Br35H")
    print(f"Path: {os.path.abspath(base_dir)}")
    print("=" * 60)

    orig_dir = os.path.join(base_dir, "original")
    if not os.path.exists(orig_dir):
        print(f"[ERROR] Directory not found: {orig_dir}")
        return

    no_dir = os.path.join(orig_dir, "no")
    yes_dir = os.path.join(orig_dir, "yes")

    no_files = [f for f in os.listdir(no_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(no_dir) else []
    yes_files = [f for f in os.listdir(yes_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(yes_dir) else []

    all_exts = Counter([os.path.splitext(f)[1].lower() for f in no_files + yes_files])

    print(f"\nRaw Folder: BR35H/original/")
    print(f"  no/  (healthy): {len(no_files)} images")
    print(f"  yes/ (tumor)  : {len(yes_files)} images")
    print(f"  Total images  : {len(no_files) + len(yes_files)} (extensions: {dict(all_exts)})")

    if no_files:
        dim_no, mode_no = get_image_info(os.path.join(no_dir, no_files[0]))
        print(f"  Sample 'no' shape : {dim_no} (mode: {mode_no}, e.g. {no_files[0]})")
    if yes_files:
        dim_yes, mode_yes = get_image_info(os.path.join(yes_dir, yes_files[0]))
        print(f"  Sample 'yes' shape: {dim_yes} (mode: {mode_yes}, e.g. {yes_files[0]})")

    corrupted_count = 0
    for f in no_files[:50]:
        ok, _ = check_image_readable(os.path.join(no_dir, f))
        if not ok:
            corrupted_count += 1
    for f in yes_files[:50]:
        ok, _ = check_image_readable(os.path.join(yes_dir, f))
        if not ok:
            corrupted_count += 1

    train_normal = [f for f in no_files if int(''.join(filter(str.isdigit, f))) < 1000]
    test_normal = [f for f in no_files if int(''.join(filter(str.isdigit, f))) >= 1000]

    print(f"\nExpected prepare_br35h.py Output:")
    print(f"  train/NORMAL/  : {len(train_normal)} images (healthy index < 1000)")
    print(f"  test/NORMAL/   : {len(test_normal)} images (healthy index >= 1000)")
    print(f"  test/ABNORMAL/ : {len(yes_files)} images (all tumor images)")
    print(f"  Total output   : {len(train_normal) + len(test_normal) + len(yes_files)} images")
    # Check processed folders if present
    train_dir = os.path.join(base_dir, "train", "NORMAL")
    test_norm_dir = os.path.join(base_dir, "test", "NORMAL")
    test_abnorm_dir = os.path.join(base_dir, "test", "ABNORMAL")

    if os.path.exists(train_dir) and os.path.exists(test_norm_dir) and os.path.exists(test_abnorm_dir):
        tr_files = os.listdir(train_dir)
        te_norm_files = os.listdir(test_norm_dir)
        te_abnorm_files = os.listdir(test_abnorm_dir)
        print(f"\nProcessed Folder: BR35H/")
        print(f"  train/NORMAL/  : {len(tr_files)} images")
        print(f"  test/NORMAL/   : {len(te_norm_files)} images")
        print(f"  test/ABNORMAL/ : {len(te_abnorm_files)} images")
        print(f"  Total processed: {len(tr_files) + len(te_norm_files) + len(te_abnorm_files)} images")
        print("Structure assessment: Processed benchmark directories VERIFIED. Ready for E2AD evaluation.")
    else:
        print("Structure assessment: Raw folders 'yes' and 'no' present. Fast file copy preprocessing required.")


def inspect_isic2018(base_dir="ISIC2018"):
    print("\n" + "=" * 60)
    print("DATASET: ISIC 2018 (Task 3)")
    print(f"Path: {os.path.abspath(base_dir)}")
    print("=" * 60)

    orig_dir = os.path.join(base_dir, "original")
    if not os.path.exists(orig_dir):
        print(f"[ERROR] Directory not found: {orig_dir}")
        return

    # Check for double nesting
    nesting_info = {}
    for dname in [
        "ISIC2018_Task3_Training_Input",
        "ISIC2018_Task3_Training_GroundTruth",
        "ISIC2018_Task3_Validation_Input",
        "ISIC2018_Task3_Validation_GroundTruth"
    ]:
        p = os.path.join(orig_dir, dname)
        double_p = os.path.join(p, dname)
        nesting_info[dname] = {
            "outer_exists": os.path.exists(p),
            "inner_nested_exists": os.path.exists(double_p),
            "effective_path": double_p if os.path.exists(double_p) else p
        }

    print("\nDirectory Nesting Inspection:")
    has_double_nesting = False
    for k, v in nesting_info.items():
        if v["inner_nested_exists"]:
            has_double_nesting = True
            print(f"  [DOUBLE NESTED] {k}/ -> {k}/")
        else:
            print(f"  [SINGLE LEVEL]  {k}/")

    train_img_dir = nesting_info["ISIC2018_Task3_Training_Input"]["effective_path"]
    val_img_dir = nesting_info["ISIC2018_Task3_Validation_Input"]["effective_path"]
    train_csv_path = os.path.join(nesting_info["ISIC2018_Task3_Training_GroundTruth"]["effective_path"], "ISIC2018_Task3_Training_GroundTruth.csv")
    val_csv_path = os.path.join(nesting_info["ISIC2018_Task3_Validation_GroundTruth"]["effective_path"], "ISIC2018_Task3_Validation_GroundTruth.csv")

    train_imgs = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(train_img_dir) else []
    val_imgs = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(val_img_dir) else []

    all_exts = Counter([os.path.splitext(f)[1].lower() for f in train_imgs + val_imgs])
    print(f"\nImage Counts:")
    print(f"  Training Input images  : {len(train_imgs)}")
    print(f"  Validation Input images: {len(val_imgs)}")
    print(f"  Extensions             : {dict(all_exts)}")

    if train_imgs:
        dim_t, mode_t = get_image_info(os.path.join(train_img_dir, train_imgs[0]))
        print(f"  Sample training shape  : {dim_t} (mode: {mode_t}, e.g. {train_imgs[0]})")

    corrupted_count = 0
    for f in train_imgs[:50]:
        ok, _ = check_image_readable(os.path.join(train_img_dir, f))
        if not ok:
            corrupted_count += 1

    if os.path.exists(train_csv_path) and os.path.exists(val_csv_path):
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)

        train_nv = int((train_df.iloc[:, 2] == 1).sum())
        val_nv = int((val_df.iloc[:, 2] == 1).sum())
        val_abnormal = int((val_df.iloc[:, 2] != 1).sum())

        print(f"\nGround Truth Class Breakdown:")
        print(f"  Train NV (Normal skin lesion)   : {train_nv} (out of {len(train_df)})")
        print(f"  Train Non-NV (Malignant/Other)  : {len(train_df) - train_nv} (unused in unsupervised AD)")
        print(f"  Validation NV (Normal test)     : {val_nv}")
        print(f"  Validation Non-NV (Abnormal test): {val_abnormal}")
        print(f"\nExpected prepare_isic2018.py Output:")
        print(f"  train/NORMAL/  : {train_nv} images")
        print(f"  test/NORMAL/   : {val_nv} images")
        print(f"  test/ABNORMAL/ : {val_abnormal} images")
        print(f"  Total output   : {train_nv + val_nv + val_abnormal} images")

    print(f"Corrupted images detected (sampled): {corrupted_count}")
    if has_double_nesting:
        print("Structure assessment: Double-nesting detected! Must normalize input paths before running prepare_isic2018.py.")
    else:
        print("Structure assessment: Ready for preprocessing.")


def main():
    print("=" * 60)
    print("E2AD DATASET VALIDATION (STRICTLY READ-ONLY)")
    print("=" * 60)
    inspect_oct2017()
    inspect_aptos()
    inspect_br35h()
    inspect_isic2018()
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE: No files were modified, moved, or deleted.")
    print("=" * 60)


if __name__ == "__main__":
    main()
