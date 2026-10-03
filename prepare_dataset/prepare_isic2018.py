import os
import random
from shutil import copyfile
import pandas as pd
import numpy as np
import argparse
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x, **k: x

random.seed(1)

parser = argparse.ArgumentParser(description='ISIC2018 Data Preparation')
parser.add_argument('--data_folder', '--data-folder', dest='data_folder',
                    default='/data/disk2T1/guoj/ISIC2018/original', type=str,
                    help='Path to raw ISIC2018 directory')
parser.add_argument('--save_folder', '--save-folder', dest='save_folder',
                    default='/data/disk2T1/guoj/ISIC2018', type=str,
                    help='Path to save preprocessed dataset')
config = parser.parse_args()

source_dir = config.data_folder
target_dir = config.save_folder

def find_file(base_dir, filename):
    # Direct candidate
    direct = os.path.join(base_dir, filename)
    if os.path.exists(direct):
        return direct
    # Recursive search (case-insensitive)
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.lower() == filename.lower():
                return os.path.join(root, f)
    raise FileNotFoundError(f"Could not find {filename} under {base_dir}")

def find_image_folder(base_dir, folder_name):
    # Pass 1: exact or case-insensitive match on basename
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root).lower() == folder_name.lower():
            jpgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(jpgs) > 0:
                return root
    # Pass 2: substring match (e.g. task3 and training/validation)
    key1 = 'training' if 'train' in folder_name.lower() else 'valid'
    for root, dirs, files in os.walk(base_dir):
        bname = os.path.basename(root).lower()
        if key1 in bname and any(k in bname for k in ['input', 'task3', 'images']):
            jpgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(jpgs) > 0:
                return root
    raise FileNotFoundError(f"Could not find folder {folder_name} containing images under {base_dir}")

train_csv_path = find_file(source_dir, 'ISIC2018_Task3_Training_GroundTruth.csv')
valid_csv_path = find_file(source_dir, 'ISIC2018_Task3_Validation_GroundTruth.csv')
train_img_dir = find_image_folder(source_dir, 'ISIC2018_Task3_Training_Input')
valid_img_dir = find_image_folder(source_dir, 'ISIC2018_Task3_Validation_Input')

print(f"[FOUND] Training CSV        : {train_csv_path}")
print(f"[FOUND] Validation CSV      : {valid_csv_path}")
print(f"[FOUND] Training Images Dir : {train_img_dir}")
print(f"[FOUND] Validation Images Dir: {valid_img_dir}")

train_csv = np.array(pd.read_csv(train_csv_path))
valid_csv = np.array(pd.read_csv(valid_csv_path))

train_normal_path = []
valid_normal_path = []
valid_abnormal_path = []

# Column index 2 corresponds to NV (Melanocytic nevus / normal class)
for line in train_csv:
    if line[2] == 1:
        train_normal_path.append(os.path.join(train_img_dir, str(line[0]) + '.jpg'))

for line in valid_csv:
    if line[2] == 1:
        valid_normal_path.append(os.path.join(valid_img_dir, str(line[0]) + '.jpg'))
    else:
        valid_abnormal_path.append(os.path.join(valid_img_dir, str(line[0]) + '.jpg'))

target_train_normal_dir = os.path.join(target_dir, 'train', 'NORMAL')
target_test_normal_dir = os.path.join(target_dir, 'test', 'NORMAL')
target_test_abnormal_dir = os.path.join(target_dir, 'test', 'ABNORMAL')

os.makedirs(target_train_normal_dir, exist_ok=True)
os.makedirs(target_test_normal_dir, exist_ok=True)
os.makedirs(target_test_abnormal_dir, exist_ok=True)

print(f"Copying {len(train_normal_path)} Train NORMAL images...")
for f in tqdm(train_normal_path, desc="Train NORMAL"):
    copyfile(f, os.path.join(target_train_normal_dir, os.path.basename(f)))

print(f"Copying {len(valid_normal_path)} Test NORMAL images...")
for f in tqdm(valid_normal_path, desc="Test NORMAL"):
    copyfile(f, os.path.join(target_test_normal_dir, os.path.basename(f)))

print(f"Copying {len(valid_abnormal_path)} Test ABNORMAL images...")
for f in tqdm(valid_abnormal_path, desc="Test ABNORMAL"):
    copyfile(f, os.path.join(target_test_abnormal_dir, os.path.basename(f)))

print("=" * 65)
print("ISIC2018 PREPROCESSING COMPLETE")
print(f"  Train NORMAL   : {len(os.listdir(target_train_normal_dir))} (Expected: 6705)")
print(f"  Test NORMAL    : {len(os.listdir(target_test_normal_dir))} (Expected: 123)")
print(f"  Test ABNORMAL  : {len(os.listdir(target_test_abnormal_dir))} (Expected: 70)")
total_copied = len(os.listdir(target_train_normal_dir)) + len(os.listdir(target_test_normal_dir)) + len(os.listdir(target_test_abnormal_dir))
print(f"  Total Processed: {total_copied} (Expected: 6898)")
print("=" * 65)
