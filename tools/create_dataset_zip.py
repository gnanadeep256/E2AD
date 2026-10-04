import os
import zipfile
import time
from tqdm.auto import tqdm

SRC_DIR = os.path.abspath("kaggle_dataset_isic")
ZIP_PATH = os.path.abspath("ISIC2018_preprocessed.zip")

print("=" * 65)
print("PACKAGING ISIC2018 PREPROCESSED DATASET ZIP")
print("=" * 65)
print(f"Source Folder : {SRC_DIR}")
print(f"Target Zip    : {ZIP_PATH}")

all_files = []
for root, dirs, files in os.walk(SRC_DIR):
    for f in files:
        full_path = os.path.join(root, f)
        rel_path = os.path.relpath(full_path, SRC_DIR)
        all_files.append((full_path, rel_path))

print(f"Total files to package: {len(all_files)}")
start_t = time.time()

# Use ZIP_STORED since JPEGs are already compressed; this makes packaging ultra-fast (~10s)
with zipfile.ZipFile(ZIP_PATH, 'w', compression=zipfile.ZIP_STORED) as zf:
    for full_p, rel_p in tqdm(all_files, desc="Archiving"):
        zf.write(full_p, arcname=rel_p)

elapsed = time.time() - start_t
sz_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
print(f"\n[SUCCESS] Packaged {len(all_files)} files into {ZIP_PATH}")
print(f"Zip Size : {sz_mb:.2f} MB ({sz_mb / 1024:.2f} GB)")
print(f"Time Taken: {elapsed:.2f} seconds")
print("=" * 65)
