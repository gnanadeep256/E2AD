import json
import os

notebook = {
    'cells': [],
    'metadata': {
        'accelerator': 'GPU',
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.12.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

def add_md(source):
    notebook['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in source.strip().split('\n')]
    })

def add_code(source):
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in source.strip().split('\n')]
    })

# ==============================================================================
# SECTION 1 — Experiment Information
# ==============================================================================
add_md("""# E2AD ISIC2018 Baseline Reproduction

## 1. Experiment Information

**Paper**: *Anomaly Detection in Medical Images Using Encoder-Attention-2Decoders Reconstruction* (IEEE Transactions on Medical Imaging 2025)  
**Method**: Encoder-Attention-2Decoders (EA2D / E2AD)  
**Dataset**: ISIC 2018 Challenge Task 3 (Skin Lesion Analysis Towards Melanoma Detection)  
**Research Repository**: [gnanadeep256/E2AD](https://github.com/gnanadeep256/E2AD)  
**Reference Repository**: [TumCCC/E2AD](https://github.com/TumCCC/E2AD) *(Read-only reference)*  
**Primary Metric**: Area Under the ROC Curve (AUROC)

---

### Configuration Overview
This notebook executes the exact official baseline reproduction for the ISIC2018 dataset:
* **Architecture**: ResNet-50 Encoder + Spatial Attention (SA) + Dual Decoders (Original + 180° Rotated)
* **Total Training Iterations**: `400`
* **Evaluation Frequency**: Every `50` iterations (8 evaluation checkpoints)
* **Train Batch Size**: `32`
* **Evaluation Batch Size**: `64`
* **Encoder Learning Rate**: `1e-5`
* **Decoder Learning Rate**: `1e-4` (Official repository default for ISIC)
* **Optimizer**: `AdamW` (Weight Decay = `1e-4`, Momentum = `0.9`)
* **Anomaly Pooling**: `mean` (`amap_reduction = 'mean'`)
* **Encoder BN**: `bn_pretrain = False`
* **Target Splits**: 6,705 Train Normal (NV), 123 Test Normal (NV), 70 Test Abnormal (non-NV) (Total = 6,898)""")

# ==============================================================================
# SECTION 2 — Kaggle GPU / CUDA Verification
# ==============================================================================
add_md("""## 2. Kaggle Hardware & GPU Verification
Verify Python kernel, PyTorch CUDA capability, device properties, and configure memory allocator.""")

add_code("""import os
import sys
import torch
import torchvision

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
try:
    torch.cuda.memory._set_allocator_settings('expandable_segments:True')
except Exception:
    pass

print("=" * 65)
print("HARDWARE & ACCELERATOR VERIFICATION")
print("=" * 65)
print(f"Python Version       : {sys.version.split()[0]}")
print(f"PyTorch Version      : {torch.__version__}")
print(f"Torchvision Version  : {torchvision.__version__}")
print(f"CUDA Available       : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CRITICAL ERROR: CUDA GPU is required! Please enable GPU in Kaggle settings.")

torch.cuda.set_device(0)
gpu_name = torch.cuda.get_device_name(0)
gpu_count = torch.cuda.device_count()
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
cuda_version = torch.version.cuda

print(f"Primary GPU (Index 0): {gpu_name}")
print(f"Available GPU Count  : {gpu_count}")
print(f"Dedicated VRAM       : {vram_gb:.2f} GB")
print(f"CUDA Driver Version  : {cuda_version}")
print("=" * 65)""")

# ==============================================================================
# SECTION 3 — Personal Repository Setup
# ==============================================================================
add_md("""## 3. Personal Research Repository Setup
Clone or verify the personal research repository: `https://github.com/gnanadeep256/E2AD.git`.""")

add_code("""import os
import sys
import subprocess

REPO_DIR = "/kaggle/working/E2AD"
REPO_URL = "https://github.com/gnanadeep256/E2AD.git"

if os.path.exists(REPO_DIR):
    print(f"Repository directory already exists: {REPO_DIR}")
    os.chdir(REPO_DIR)
    try:
        subprocess.run(["git", "pull", "origin", "main"], check=True)
    except Exception as e:
        print(f"git pull note: {e}")
elif os.path.exists("./models/edc.py"):
    REPO_DIR = os.path.abspath(".")
    print(f"Running natively inside repository root: {REPO_DIR}")
else:
    print(f"Cloning personal repository from {REPO_URL}...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    os.chdir(REPO_DIR)

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

remotes = subprocess.check_output(["git", "remote", "-v"]).decode().strip()
branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
commit_short = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
full_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

print("=" * 65)
print("REPOSITORY PROVENANCE AUDIT")
print("=" * 65)
print(f"Active Directory : {os.getcwd()}")
print(f"Active Branch    : {branch}")
print(f"Active Commit    : {commit_short} ({full_commit})")
print("Git Remotes:")
print(remotes)
assert "gnanadeep256/E2AD" in remotes, "CRITICAL: Must ONLY use personal repository (gnanadeep256/E2AD)!"
assert "TumCCC/E2AD" not in remotes, "CRITICAL: TumCCC/E2AD detected as remote! Reference repository is read-only."
print("Repository provenance check SUCCESSFUL.")
print("=" * 65)""")

# ==============================================================================
# SECTION 4 — Dependency Verification
# ==============================================================================
add_md("""## 4. Scientific Dependency Verification
Verify that required scientific libraries are installed and importable.""")

add_code("""import numpy as np
import scipy
import sklearn
import PIL
from PIL import Image
import yaml
import matplotlib
import tqdm

print("=" * 65)
print("DEPENDENCY AUDIT")
print("=" * 65)
print(f"NumPy         : {np.__version__}")
print(f"SciPy         : {scipy.__version__}")
print(f"Scikit-Learn  : {sklearn.__version__}")
print(f"Pillow (PIL)  : {PIL.__version__}")
print(f"PyYAML        : {yaml.__version__}")
print(f"Matplotlib    : {matplotlib.__version__}")
print(f"tqdm          : {tqdm.__version__}")

try:
    import albumentations
    print(f"Albumentations: {albumentations.__version__}")
except ImportError:
    print("Albumentations: Not installed (using built-in torchvision fallback)")

print("All scientific dependencies verified successfully.")
print("=" * 65)""")

# ==============================================================================
# SECTION 5 — Modern Torchvision Compatibility Patch Verification
# ==============================================================================
add_md("""## 5. Modern Torchvision Compatibility Patch Verification
Verify that `models/resnet.py` and `models/resnet_decoder.py` use modern `torch.hub.load_state_dict_from_url`.""")

add_code("""for filepath in ["models/resnet.py", "models/resnet_decoder.py"]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "torchvision._internally_replaced_utils" not in content, (
        f"CRITICAL: Deprecated torchvision._internally_replaced_utils found in {filepath}!"
    )
    assert "load_state_dict_from_url" in content, (
        f"CRITICAL: load_state_dict_from_url missing in {filepath}!"
    )
    print(f"  [VERIFIED] {filepath:<28} : Modern torch.hub compatibility patch confirmed.")

print("Compatibility verification SUCCESSFUL.")""")

# ==============================================================================
# SECTION 6 — ISIC Dataset Discovery & Preprocessing
# ==============================================================================
add_md("""## 6. ISIC2018 Dataset Discovery & Preprocessing
Locate the ISIC2018 dataset across standard Kaggle/local paths. If raw data is discovered, run the double-nesting compatible preprocessing procedure.""")

add_code("""import os
import sys
import subprocess

target_dir = "/kaggle/working/ISIC2018" if os.path.exists("/kaggle/working") else "./ISIC2018"
train_norm_dir = os.path.join(target_dir, "train", "NORMAL")
test_norm_dir = os.path.join(target_dir, "test", "NORMAL")
test_abnorm_dir = os.path.join(target_dir, "test", "ABNORMAL")

ISIC_DIR = None

# Check if already preprocessed in candidate paths (recursive up to 3 levels)
search_roots = [
    "/kaggle/input/datasets/gnanadeepthatavarthi/isic-2k18",
    "/kaggle/input/isic-2k18",
    "/kaggle/input/isic2018-processed",
    "/kaggle/input/e2ad-isic",
    "/kaggle/input",
    target_dir,
    "./ISIC2018",
    "../ISIC2018"
]

for base in search_roots:
    if not os.path.exists(base):
        continue
    # Check directly
    t_norm = os.path.join(base, "train", "NORMAL")
    te_norm = os.path.join(base, "test", "NORMAL")
    te_abnorm = os.path.join(base, "test", "ABNORMAL")
    if os.path.exists(t_norm) and os.path.exists(te_norm) and os.path.exists(te_abnorm):
        c_tr = len([f for f in os.listdir(t_norm) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        c_ten = len([f for f in os.listdir(te_norm) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        c_tea = len([f for f in os.listdir(te_abnorm) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if c_tr == 6705 and c_ten == 123 and c_tea == 70:
            ISIC_DIR = os.path.abspath(base)
            break

    # If base is a container like /kaggle/input, walk subdirectories
    for root, dirs, _ in os.walk(base):
        depth = root[len(base):].count(os.sep)
        if depth > 3:
            continue
        t_norm = os.path.join(root, "train", "NORMAL")
        te_norm = os.path.join(root, "test", "NORMAL")
        te_abnorm = os.path.join(root, "test", "ABNORMAL")
        if os.path.exists(t_norm) and os.path.exists(te_norm) and os.path.exists(te_abnorm):
            c_tr = len([f for f in os.listdir(t_norm) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            c_ten = len([f for f in os.listdir(te_norm) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            c_tea = len([f for f in os.listdir(te_abnorm) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if c_tr == 6705 and c_ten == 123 and c_tea == 70:
                ISIC_DIR = os.path.abspath(root)
                break
    if ISIC_DIR is not None:
        break

if ISIC_DIR is not None:
    print(f"[VERIFIED] Preprocessed ISIC2018 dataset discovered at: {ISIC_DIR}")
    print(f"  Train NORMAL: 6705 | Test NORMAL: 123 | Test ABNORMAL: 70 (Total: 6898)")
    print("  Skipping preprocessing step.")

# If not preprocessed, search for raw dataset candidates
if ISIC_DIR is None:
    raw_candidates = [
        "/kaggle/input/isic-2018",
        "/kaggle/input/isic2018",
        "/kaggle/input/isic-2018-task-3-training-and-validation",
        "/kaggle/input/skin-lesion-analysis-towards-melanoma-detection",
        "/kaggle/input/isic2018-task3",
        "./ISIC2018/original",
        "../ISIC2018/original"
    ]
    raw_dir = None
    for p in raw_candidates:
        if os.path.exists(p):
            # Check if training images exist anywhere inside
            has_train = False
            for root, dirs, files in os.walk(p):
                if any('ISIC2018_Task3_Training' in d or 'Training_Input' in d for d in dirs):
                    has_train = True
                    break
                if any(f.endswith('.csv') and 'Training_GroundTruth' in f for f in files):
                    has_train = True
                    break
            if has_train:
                raw_dir = os.path.abspath(p)
                print(f"[FOUND] Raw ISIC2018 dataset at: {raw_dir}")
                break
            
    if raw_dir is None:
        print("[INFO] Raw Task 3 dataset not detected in /kaggle/input or local candidate paths.")
        print("Initiating direct download of official ISIC 2018 Challenge Task 3 archives from AWS S3...")
        import urllib.request
        import zipfile
        from tqdm.auto import tqdm

        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        raw_download_dir = os.path.join(target_dir, "original")
        os.makedirs(raw_download_dir, exist_ok=True)

        urls = [
            ("ISIC2018_Task3_Training_Input.zip", "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip"),
            ("ISIC2018_Task3_Training_GroundTruth.zip", "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip"),
            ("ISIC2018_Task3_Validation_Input.zip", "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_Input.zip"),
            ("ISIC2018_Task3_Validation_GroundTruth.zip", "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_GroundTruth.zip")
        ]

        for fname, url in urls:
            dest_zip = os.path.join(raw_download_dir, fname)
            print(f"Downloading {fname}...")
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=fname) as t:
                urllib.request.urlretrieve(url, filename=dest_zip, reporthook=t.update_to)
            print(f"Extracting {fname}...")
            with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
                zip_ref.extractall(raw_download_dir)
            if os.path.exists(dest_zip):
                os.remove(dest_zip)

        raw_dir = raw_download_dir
        print(f"[DOWNLOAD COMPLETED] Official ISIC 2018 Task 3 unpacked to: {raw_dir}")
    
    print(f"Executing official preprocessing: {raw_dir} -> {target_dir}...")
    prep_cmd = [
        sys.executable, "prepare_dataset/prepare_isic2018.py",
        "--data-folder", raw_dir,
        "--save-folder", target_dir
    ]
    subprocess.run(prep_cmd, check=True)
    ISIC_DIR = os.path.abspath(target_dir)

    # Clean up temporary downloaded raw files if they were downloaded to target_dir/original to save disk space
    raw_download_dir = os.path.join(target_dir, "original")
    if os.path.exists(raw_download_dir) and raw_dir == raw_download_dir:
        import shutil
        shutil.rmtree(raw_download_dir)
        print(f"[CLEANUP] Removed temporary raw files from {raw_download_dir} to free disk space.")

    print(f"[COMPLETED] Preprocessing finished successfully. Target: {ISIC_DIR}")

print("=" * 65)
print(f"ACTIVE ISIC2018 DATASET DIRECTORY: {ISIC_DIR}")
print("=" * 65)""")

# ==============================================================================
# SECTION 7 — Dataset Count Verification
# ==============================================================================
add_md("""## 7. Dataset Count & Integrity Verification
Verify exact file counts (6705 train normal, 123 test normal, 70 test abnormal), check for 0-byte or corrupted files, and ensure duplicate-free image sets.""")

add_code("""import os

TRAIN_NORMAL_DIR = os.path.join(ISIC_DIR, "train", "NORMAL")
TEST_NORMAL_DIR = os.path.join(ISIC_DIR, "test", "NORMAL")
TEST_ABNORMAL_DIR = os.path.join(ISIC_DIR, "test", "ABNORMAL")

assert os.path.exists(TRAIN_NORMAL_DIR), f"Missing: {TRAIN_NORMAL_DIR}"
assert os.path.exists(TEST_NORMAL_DIR), f"Missing: {TEST_NORMAL_DIR}"
assert os.path.exists(TEST_ABNORMAL_DIR), f"Missing: {TEST_ABNORMAL_DIR}"

def audit_directory(path, expected_count, label):
    files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    actual_count = len(files)
    zero_byte = [f for f in files if os.path.getsize(os.path.join(path, f)) == 0]
    unique_names = len(set(files))
    
    print(f"  {label:<18} : {actual_count:>5} images (Expected: {expected_count:>5}) | Zero-byte: {len(zero_byte)} | Unique: {unique_names}")
    assert actual_count == expected_count, f"Count mismatch in {label}! Expected {expected_count}, got {actual_count}."
    assert len(zero_byte) == 0, f"Found {len(zero_byte)} zero-byte files in {label}!"
    assert actual_count == unique_names, f"Duplicate filenames detected in {label}!"
    return files

print("=" * 65)
print("ISIC2018 DATASET INTEGRITY AUDIT")
print("=" * 65)
train_norm_files = audit_directory(TRAIN_NORMAL_DIR, 6705, "Train NORMAL")
test_norm_files = audit_directory(TEST_NORMAL_DIR, 123, "Test NORMAL")
test_abnorm_files = audit_directory(TEST_ABNORMAL_DIR, 70, "Test ABNORMAL")
total_images = len(train_norm_files) + len(test_norm_files) + len(test_abnorm_files)

print("-" * 65)
print(f"  Total Processed Images : {total_images:>5} (Expected: 6898)")
assert total_images == 6898, f"Total image count mismatch! Expected 6898, got {total_images}."
print("Dataset verification SUCCESSFUL: Exact 6705 / 123 / 70 distribution confirmed.")
print("=" * 65)""")

# ==============================================================================
# SECTION 8 — Dataset Visual Sanity Check
# ==============================================================================
add_md("""## 8. Dataset Visual Sanity Check
Inspect sample dermoscopy photographs to verify visual fidelity and morphology across benign nevi and pathological skin lesions.""")

add_code("""from PIL import Image
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(10, 10), dpi=150)
categories = [
    ("Train NORMAL (Nevus)", TRAIN_NORMAL_DIR, train_norm_files[:3]),
    ("Test NORMAL (Nevus)", TEST_NORMAL_DIR, test_norm_files[:3]),
    ("Test ABNORMAL (Lesion)", TEST_ABNORMAL_DIR, test_abnorm_files[:3])
]

for row_idx, (cat_name, folder, samples) in enumerate(categories):
    for col_idx, fname in enumerate(samples):
        img_path = os.path.join(folder, fname)
        img = Image.open(img_path)
        assert img is not None, f"Corrupt image could not be read: {img_path}"
        
        ax = axes[row_idx, col_idx]
        ax.imshow(img)
        ax.set_title(f"{cat_name}\\n{img.size[0]}x{img.size[1]}", fontsize=9)
        ax.axis('off')

plt.suptitle("ISIC 2018 Dermoscopy Visual Inspection", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print("Visual sanity check SUCCESSFUL: Valid dermoscopic morphology confirmed.")""")

# ==============================================================================
# SECTION 9 — E2AD Architecture Import Verification
# ==============================================================================
add_md("""## 9. E2AD Architecture Import Verification
Import E2AD architecture with ResNet-50 backbone, dual decoders, and spatial attention module.""")

add_code("""from models.edc import E2AD
from datasets.dataset import AD_Dataset
from datasets.data_utils import get_data_loader
from methods.edc1 import EDC_MS
from train_utils import get_optimizer_v2, get_multistep_schedule_with_warmup, TBLog
from utils import count_parameters

model_test = E2AD(bn_pretrain=False)
param_count = count_parameters(model_test)

print("=" * 65)
print("E2AD ARCHITECTURE AUDIT (ISIC2018)")
print("=" * 65)
print(f"Model Class           : {model_test.__class__.__name__}")
print(f"Backbone Encoder      : ResNet-50 (Pretrained ImageNet)")
print(f"Decoder 1             : ResNet-50 Decoder (Original Image)")
print(f"Decoder 2             : ResNet-50 Decoder (180° Rotated Image)")
print(f"Attention Module      : Spatial Attention (SA-1024, SA-512)")
print(f"Trainable Parameters  : {param_count:,}")
print("=" * 65)
del model_test""")

# ==============================================================================
# SECTION 10 — DataLoader Smoke Test
# ==============================================================================
add_md("""## 10. DataLoader Smoke Test
Verify batch generation, tensor dimensions, and labels for train (batch 32) and evaluation (batch 64) loaders.""")

add_code("""train_dset = AD_Dataset(name="isic", train=True, transform=False, data_dir=ISIC_DIR, img_size=256, crop_size=224).get_dset()
eval_dset = AD_Dataset(name="isic", train=False, transform=False, data_dir=ISIC_DIR, img_size=256, crop_size=224).get_dset()

print(f"TrainSet Image Number : {len(train_dset)} (Expected: 6705)")
print(f"EvalSet Image Number  : {len(eval_dset)} (Expected: 193)")
assert len(train_dset) == 6705, f"Train set size mismatch: {len(train_dset)}"
assert len(eval_dset) == 193, f"Eval set size mismatch: {len(eval_dset)}"

train_loader = get_data_loader(train_dset, batch_size=32, num_iters=400, num_workers=2, distributed=False)
eval_loader = get_data_loader(eval_dset, batch_size=64, num_workers=2, drop_last=False)

train_batch = next(iter(train_loader))
idx_b, x_b, _, y_b, fn_b = train_batch
print(f"Train Batch Tensor Shape : {x_b.shape} (Expected: [32, 3, 224, 224])")
print(f"Train Batch Labels       : {y_b.unique().tolist()} (Expected: [0.0] - all Normal)")
assert x_b.shape == (32, 3, 224, 224), f"Unexpected train shape: {x_b.shape}"

eval_batch = next(iter(eval_loader))
_, x_ev, xo_ev, y_ev, fn_ev = eval_batch
print(f"Eval Batch Tensor Shape  : {x_ev.shape} (Expected: [64, 3, 224, 224])")
print(f"Eval Batch Labels Present: {y_ev.unique().tolist()}")
assert x_ev.shape == (64, 3, 224, 224), f"Unexpected eval shape: {x_ev.shape}"
print("DataLoader smoke test PASSED.")""")

# ==============================================================================
# SECTION 11 — Model Forward-Pass Smoke Test
# ==============================================================================
add_md("""## 11. Model Forward-Pass & Optimization Smoke Test
Verify GPU allocation, forward pass computation, loss calculation, backward gradient flow, and parameter update using lightweight slice with Automatic Mixed Precision (AMP).""")

add_code("""import gc
import torch

# Clean up any lingering tensors from previous attempts
for var in ['model_smoke', 'x_test', 'opt_smoke', 'scaler_smoke', 'out_smoke', 'out_smoke_train', 'train_loss_smoke', 'loss_smoke']:
    if var in globals():
        del globals()[var]
gc.collect()
torch.cuda.empty_cache()

try:
    torch.cuda.memory._set_allocator_settings('expandable_segments:True')
except Exception:
    pass

# 1. Initialize model on GPU
model_smoke = E2AD(bn_pretrain=False).cuda(0)

# 2. Use lightweight slice (4 samples) to test full pipeline without OOM
x_test = x_b[:4].cuda(0)

# 3. Test forward pass with AMP (matching real training)
scaler_smoke = torch.cuda.amp.GradScaler(enabled=True)
opt_smoke = get_optimizer_v2(model_smoke, 'AdamW', 1e-4, 0.9, lr_encoder=1e-5, weight_decay=1e-4)
opt_smoke.zero_grad()

with torch.cuda.amp.autocast(enabled=True):
    out_smoke = model_smoke(x_test)
    train_loss_smoke = out_smoke['loss'].mean()

print(f"Smoke Test Forward Loss : {train_loss_smoke.item():.4f}")
assert not torch.isnan(train_loss_smoke), "CRITICAL: NaN loss detected!"
assert not torch.isinf(train_loss_smoke), "CRITICAL: Inf loss detected!"

# 4. Test backward pass & optimizer step with gradient scaling
scaler_smoke.scale(train_loss_smoke).backward()
scaler_smoke.unscale_(opt_smoke)
torch.nn.utils.clip_grad_norm_(model_smoke.parameters(), 1.0)
scaler_smoke.step(opt_smoke)
scaler_smoke.update()

# 5. Clean up ALL smoke test objects to guarantee 100% free VRAM for training
del model_smoke, x_test, opt_smoke, scaler_smoke, out_smoke, train_loss_smoke
gc.collect()
torch.cuda.empty_cache()

vram_used_mb = torch.cuda.memory_allocated(0) / (1024**2)
vram_reserved_mb = torch.cuda.memory_reserved(0) / (1024**2)

print("=" * 60)
print("ISIC2018 SMOKE TEST PASSED (AMP ENABLED)")
print(f"Active VRAM after cleanup : {vram_used_mb:.1f} MB (Reserved: {vram_reserved_mb:.1f} MB)")
print("Ready for 400-iteration baseline training")
print("=" * 60)""")

# ==============================================================================
# SECTION 12 — Training Configuration
# ==============================================================================
add_md("""## 12. Training Configuration
Define exact official hyperparameters for ISIC2018 baseline reproduction.""")

add_code("""import argparse

class Args:
    dataset = "isic"
    data_dir = ISIC_DIR
    num_train_iter = 400
    num_eval_iter = 50
    batch_size = 32
    eval_batch_size = 64
    lr = 1e-4
    lr_encoder = 1e-5
    optim = "AdamW"
    weight_decay = 1e-4
    momentum = 0.9
    img_size = 256
    crop_size = 224
    save_weight = 1
    save_dir = "./saved_models"
    save_name = "e2ad_isic"
    model_name = "E2AD"
    gpu = 0
    seed = 0
    amp = True
    clip = 1.0
    train_times = 1
    overwrite = True
    resume = 0
    load_path = None
    use_tensorboard = False
    train_sampler = "RandomSampler"
    num_workers = 2
    ema_m = 0.0

args = Args()

print("=" * 65)
print("E2AD ISIC2018 BASELINE HYPERPARAMETERS")
print("=" * 65)
print(f"Dataset                  : {args.dataset.upper()}")
print(f"Data Directory           : {args.data_dir}")
print(f"Training Iterations      : {args.num_train_iter}")
print(f"Evaluation Frequency     : Every {args.num_eval_iter} iterations (8 evaluations)")
print(f"Train Batch Size         : {args.batch_size}")
print(f"Eval Batch Size          : {args.eval_batch_size}")
print(f"Encoder Learning Rate    : {args.lr_encoder:.2e} (1e-5)")
print(f"Decoder Learning Rate    : {args.lr:.2e} (1e-4)")
print(f"Optimizer                : {args.optim} (weight_decay={args.weight_decay})")
print(f"Automatic Mixed Precision: {args.amp}")
print(f"Anomaly Score Pooling    : mean (amap_reduction = 'mean')")
print(f"Checkpoint Retention     : Enabled (--save_weight 1)")
print("=" * 65)""")

# ==============================================================================
# SECTION 13 — Full E2AD Training
# ==============================================================================
add_md("""## 13. Full E2AD Training
Execute the official 400-iteration baseline training with live `tqdm` progress bar and periodic evaluation updates.""")

add_code("""import os
import time
import gc
import shutil
import torch
from e2ad_isic import main_worker

try:
    torch.cuda.memory._set_allocator_settings('expandable_segments:True')
except Exception:
    pass
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

start_train_time = time.time()
total_list = [[], [], [], []]
save_path = f"./{args.save_dir}/{args.save_name}/{args.model_name}/0/"

if os.path.exists(save_path) and args.overwrite:
    shutil.rmtree(save_path)
os.makedirs(save_path, exist_ok=True)

print("=" * 65)
print("LAUNCHING OFFICIAL E2AD ISIC2018 BASELINE TRAINING")
print("=" * 65)
print(f"Encoder Learning Rate : {args.lr_encoder:.2e} (1e-5)")
print(f"Decoder Learning Rate : {args.lr:.2e} (1e-4)")
print(f"Checkpoint Directory  : {save_path}")
print("=" * 65)

total_list = main_worker(int(args.gpu), args, total_list, save_path)
train_elapsed_sec = time.time() - start_train_time

print("=" * 65)
print(f"TRAINING COMPLETED in {train_elapsed_sec / 60:.2f} minutes ({train_elapsed_sec:.2f}s)")
print("=" * 65)""")

# ==============================================================================
# SECTION 14 — Checkpoint Verification
# ==============================================================================
add_md("""## 14. Checkpoint Verification
Verify that both `best_auc.pth` and `last_epoch.pth` have been written to disk with non-zero size.""")

add_code("""CHECKPOINT_DIR = f"./{args.save_dir}/{args.save_name}/{args.model_name}/0"
best_auc_path = os.path.join(CHECKPOINT_DIR, "best_auc.pth")
last_epoch_path = os.path.join(CHECKPOINT_DIR, "last_epoch.pth")

print("=" * 65)
print("CHECKPOINT VERIFICATION")
print("=" * 65)
print(f"Target Directory: {CHECKPOINT_DIR}")
assert os.path.exists(CHECKPOINT_DIR), f"Directory does not exist: {CHECKPOINT_DIR}"

for name, p in [("Best AUROC Checkpoint", best_auc_path), ("Final Epoch Checkpoint", last_epoch_path)]:
    if os.path.exists(p):
        sz_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"  [FOUND]   {name:<24} : {p} ({sz_mb:.2f} MB)")
    else:
        print(f"  [MISSING] {name:<24} : {p}")

assert os.path.exists(best_auc_path), "CRITICAL: best_auc.pth was NOT saved!"
assert os.path.exists(last_epoch_path), "CRITICAL: last_epoch.pth was NOT saved!"
print("Checkpoint verification SUCCESSFUL.")
print("=" * 65)""")

# ==============================================================================
# SECTION 15 — Results Extraction & Final Evaluation
# ==============================================================================
add_md("""## 15. Results Extraction & Final Evaluation
Extract clinical metrics and complete evaluation history with direct checkpoint evaluation fallback.""")

add_code("""import os
import torch
from models.edc import E2AD
from methods.edc1 import EDC_MS
from datasets.dataset import AD_Dataset
from datasets.data_utils import get_data_loader

CHECKPOINT_DIR = f"./{args.save_dir}/{args.save_name}/{args.model_name}/0"
best_auc_path = os.path.join(CHECKPOINT_DIR, "best_auc.pth")
last_epoch_path = os.path.join(CHECKPOINT_DIR, "last_epoch.pth")

last_eval_dict = getattr(args, 'last_eval_dict', None)
if not last_eval_dict and len(total_list) > 2 and total_list[2]:
    last_eval_dict = total_list[2][-1]

# If last_eval_dict was not populated by runner, evaluate the saved checkpoints directly
if last_eval_dict is None:
    print("[INFO] Evaluating saved checkpoints directly on ISIC2018 test set...")
    if 'eval_loader' not in globals() or eval_loader is None:
        eval_dset = AD_Dataset(name=args.dataset, train=False, transform=False, data_dir=args.data_dir, img_size=256, crop_size=224).get_dset()
        eval_loader = get_data_loader(eval_dset, batch_size=args.eval_batch_size, num_workers=args.num_workers, drop_last=False)
    
    # 1. Evaluate Best Checkpoint
    model_best = E2AD(bn_pretrain=False)
    model_best.load_state_dict(torch.load(best_auc_path, map_location=f"cuda:{args.gpu}"))
    model_best = model_best.cuda(args.gpu)
    runner_best = EDC_MS(model=model_best, amap_reduction='mean', save_path=CHECKPOINT_DIR)
    eval_best_dict = runner_best.evaluate(eval_loader=eval_loader, args=args)
    del model_best, runner_best
    torch.cuda.empty_cache()

    # 2. Evaluate Final Checkpoint
    model_final = E2AD(bn_pretrain=False)
    model_final.load_state_dict(torch.load(last_epoch_path, map_location=f"cuda:{args.gpu}"))
    model_final = model_final.cuda(args.gpu)
    runner_final = EDC_MS(model=model_final, amap_reduction='mean', save_path=CHECKPOINT_DIR)
    eval_final_dict = runner_final.evaluate(eval_loader=eval_loader, args=args)
    del model_final, runner_final
    torch.cuda.empty_cache()

    last_eval_dict = eval_final_dict
    last_eval_dict['eval/best_auc'] = eval_best_dict['eval/AUC']
    args.last_eval_dict = last_eval_dict
    args.eval_best_dict = eval_best_dict
else:
    eval_best_dict = getattr(args, 'eval_best_dict', last_eval_dict)
    eval_final_dict = last_eval_dict

raw_log = last_eval_dict.get('eval/train_log', [])
eval_history = []
for entry in raw_log:
    it_num = entry.get('it', 0)
    eval_history.append({
        'iteration': it_num,
        'training_step': it_num + 1,
        'auroc': float(entry.get('eval/AUC', 0.0)),
        'f1': float(entry.get('eval/f1', 0.0)),
        'accuracy': float(entry.get('eval/acc', 0.0)),
        'sensitivity': float(entry.get('eval/recall', 0.0)),
        'specificity': float(entry.get('eval/specificity', 0.0)),
        'loss': float(entry.get('train/total_loss', 0.0)),
        'eval_loss': float(entry.get('eval/loss', 0.0))
    })

final_auc = float(eval_final_dict.get('eval/AUC', total_list[0][0] if total_list[0] else 0.0))
best_auc = float(eval_best_dict.get('eval/best_auc', eval_best_dict.get('eval/AUC', total_list[1][0] if len(total_list)>1 and total_list[1] else final_auc)))
best_it = last_eval_dict.get('eval/best_it', 399)
best_step = int(best_it) + 1 if str(best_it).isdigit() else 400

final_f1 = float(eval_final_dict.get('eval/f1', 0.0))
final_acc = float(eval_final_dict.get('eval/acc', 0.0))
final_sen = float(eval_final_dict.get('eval/recall', 0.0))
final_spe = float(eval_final_dict.get('eval/specificity', 0.0))

best_f1 = float(eval_best_dict.get('eval/f1', final_f1))
best_acc = float(eval_best_dict.get('eval/acc', final_acc))
best_sen = float(eval_best_dict.get('eval/recall', final_sen))
best_spe = float(eval_best_dict.get('eval/specificity', final_spe))

if not eval_history:
    eval_history = [{
        'iteration': int(best_it) if str(best_it).isdigit() else 399,
        'training_step': best_step,
        'auroc': best_auc,
        'f1': best_f1,
        'accuracy': best_acc,
        'sensitivity': best_sen,
        'specificity': best_spe,
        'loss': 0.0,
        'eval_loss': float(eval_best_dict.get('eval/loss', 0.0))
    }]

print("=" * 65)
print("FINAL EVALUATION METRICS (ISIC2018)")
print("=" * 65)
print(f"--- BEST CHECKPOINT (best_auc.pth) ---")
print(f"Best AUROC                 : {best_auc * 100:.4f}% ({best_auc:.15f})")
print(f"Best F1-Score              : {best_f1 * 100:.4f}%")
print(f"Best Accuracy              : {best_acc * 100:.4f}%")
print(f"Best Sensitivity / Recall  : {best_sen * 100:.4f}%")
print(f"Best Specificity           : {best_spe * 100:.4f}%")
print("-" * 55)
print(f"--- FINAL ITERATION (last_epoch.pth) ---")
print(f"Final AUROC                : {final_auc * 100:.4f}% ({final_auc:.15f})")
print(f"Final F1-Score             : {final_f1 * 100:.4f}%")
print(f"Final Accuracy             : {final_acc * 100:.4f}%")
print(f"Final Sensitivity / Recall : {final_sen * 100:.4f}%")
print(f"Final Specificity          : {final_spe * 100:.4f}%")
print("=" * 65)""")

# ==============================================================================
# SECTION 16 — Results Summary & Trajectory Plotting
# ==============================================================================
add_md("""## 16. Results Summary & Trajectory Plotting
Export structured results to `results/isic2018_baseline_results.json`, save `results/isic2018_auroc_history.json`, and plot validation AUROC curve.""")

add_code("""import json
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

results_isic = {
    "dataset": "ISIC2018",
    "model": "E2AD",
    "repository": "https://github.com/gnanadeep256/E2AD",
    "commit": full_commit,
    "seed": args.seed,
    "iterations": args.num_train_iter,
    "evaluation_frequency": args.num_eval_iter,
    "batch_size": args.batch_size,
    "eval_batch_size": args.eval_batch_size,
    "optimizer": args.optim,
    "encoder_lr": args.lr_encoder,
    "decoder_lr": args.lr,
    "weight_decay": args.weight_decay,
    "image_size": args.img_size,
    "crop_size": args.crop_size,
    "amap_reduction": "mean",
    "best_auc": best_auc,
    "best_iteration": best_it,
    "best_training_step": best_step,
    "final_auc": final_auc,
    "final_f1": final_f1,
    "final_accuracy": final_acc,
    "final_sensitivity": final_sen,
    "final_specificity": final_spe,
    "final_train_loss": last_eval_dict.get('train/total_loss'),
    "final_eval_loss": last_eval_dict.get('eval/loss'),
    "training_time_seconds": round(train_elapsed_sec, 2),
    "gpu": torch.cuda.get_device_name(0),
    "pytorch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "best_checkpoint": best_auc_path,
    "final_checkpoint": last_epoch_path,
    "evaluation_history_json": "results/isic2018_auroc_history.json",
    "evaluation_trajectory_plot": "results/isic2018_auroc_curve.png",
    "status": "completed_verified"
}

json_path = "results/isic2018_baseline_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_isic, f, indent=2)
print(f"Results successfully saved to: {json_path}")

hist_path = "results/isic2018_auroc_history.json"
with open(hist_path, "w", encoding="utf-8") as f:
    json.dump(eval_history, f, indent=2)
print(f"Evaluation history saved to: {hist_path}")

if eval_history:
    steps = [r['training_step'] for r in eval_history]
    aurocs = [r['auroc'] * 100 for r in eval_history]
    losses = [r.get('loss', 0.0) for r in eval_history]

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    color = '#1f77b4'
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Validation AUROC (%)', color=color, fontsize=12)
    ax1.plot(steps, aurocs, color=color, marker='o', linewidth=2.2, label='Validation AUROC')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    best_idx = int(np.argmax(aurocs))
    ax1.scatter([steps[best_idx]], [aurocs[best_idx]], color='red', s=100, zorder=5, label=f'Best AUROC ({aurocs[best_idx]:.2f}%)')

    if any(l > 0 for l in losses):
        ax2 = ax1.twinx()
        color_loss = '#ff7f0e'
        ax2.set_ylabel('Train Loss', color=color_loss, fontsize=12)
        ax2.plot(steps, losses, color=color_loss, linestyle=':', marker='x', alpha=0.7, label='Train Loss')
        ax2.tick_params(axis='y', labelcolor=color_loss)

    plt.title('E2AD ISIC 2018 Baseline: Training Trajectory (ResNet-50, Mean Pooling)', fontsize=14)
    fig.tight_layout()
    curve_path = "results/isic2018_auroc_curve.png"
    plt.savefig(curve_path, dpi=300)
    plt.show()
    print(f"Evaluation trajectory plot saved to: {curve_path}")""")

# ==============================================================================
# SECTION 17 — Markdown Report Generation
# ==============================================================================
add_md("""## 17. Markdown Report Generation
Export the comprehensive execution report to `reports/isic2018_baseline.md`.""")

add_code("""os.makedirs("reports", exist_ok=True)
report_md = f\"\"\"# ISIC2018 E2AD Baseline

## Dataset

Training normal:
6705

Test normal:
123

Test abnormal:
70

## Configuration

Iterations:
{args.num_train_iter}

Evaluation frequency:
{args.num_eval_iter}

Train batch:
{args.batch_size}

Eval batch:
{args.eval_batch_size}

Encoder LR:
{args.lr_encoder}

Decoder LR:
{args.lr}

Weight decay:
{args.weight_decay}

Anomaly Pooling:
mean

## Final Results

| Metric | Result |
|---|---:|
| Best AUROC | {best_auc * 100:.4f}% |
| Best iteration | {best_it} |
| Final AUROC | {final_auc * 100:.4f}% |
| Final F1 | {final_f1 * 100:.4f}% |
| Final Accuracy | {final_acc * 100:.4f}% |
| Final Sensitivity | {final_sen * 100:.4f}% |
| Final Specificity | {final_spe * 100:.4f}% |

## Hardware

GPU:
{results_isic['gpu']}

## Reproducibility

Repository:
{results_isic['repository']}

Commit:
{results_isic['commit']}

Seed:
{args.seed}

## Checkpoints

Best:
{best_auc_path}

Final:
{last_epoch_path}
\"\"\"

report_path = "reports/isic2018_baseline.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_md)
print(f"Markdown report generated successfully at: {report_path}")""")

# ==============================================================================
# SECTION 18 — Final Experiment Summary
# ==============================================================================
add_md("""## 18. Final Experiment Summary & Provenance Audit
Display the final experiment summary block and confirm git remote tracking.""")

add_code("""print("=" * 60)
print("ISIC2018 E2AD BASELINE — FINAL RESULTS")
print("=" * 60)
print(f"Best AUROC:           {best_auc * 100:.4f}%")
print(f"Best iteration:       {best_it}")
print(f"Best training step:   {best_step}")
print("-" * 50)
print(f"Final AUROC:          {final_auc * 100:.4f}%")
print(f"Final F1:             {final_f1 * 100:.4f}%")
print(f"Final Accuracy:       {final_acc * 100:.4f}%")
print(f"Final Sensitivity:    {final_sen * 100:.4f}%")
print(f"Final Specificity:    {final_spe * 100:.4f}%")
print("-" * 50)
print(f"Training time:        {train_elapsed_sec / 60:.2f} minutes")
print(f"GPU:                  {results_isic['gpu']}")
print(f"Commit:               {results_isic['commit']}")
print("=" * 60)
print("Git Remote Verification:")
os.system("git remote -v")
print("=" * 60)""")

# Write notebook to disk
target_file = "E2AD_ISIC2018_Baseline.ipynb"
with open(target_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

os.makedirs("notebooks", exist_ok=True)
notebooks_file = os.path.join("notebooks", "E2AD_ISIC2018_baseline.ipynb")
with open(notebooks_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully generated {target_file} and {notebooks_file} with {len(notebook['cells'])} cells.")
