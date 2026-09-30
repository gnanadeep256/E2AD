# Source-Code Compatibility Audit: E2AD BR35H Baseline on Modern Kaggle GPU

**Date**: September 29, 2026  
**Target Environment**: Kaggle Notebook / Script with NVIDIA GPU (Tesla T4 16GB or P100 16GB)  
**Target Software Stack**: Linux (Ubuntu 22.04), Python 3.10 / 3.11, PyTorch 2.1+ / 2.4+ (CUDA 11.8 / 12.1), Torchvision 0.16+  
**Benchmark Target**: BR35H Brain Tumor Anomaly Detection (1,000 train normal, 500 test normal, 1,500 test abnormal)  

---

## 1. PyTorch & Torchvision Compatibility Matrix

A thorough line-by-line inspection of all imports and API invocations across the repository revealed the following compatibility profile between the original development environment (`Python 3.8.12`, `torch==1.12.0+cu113`, `torchvision==0.13.0+cu113`) and modern Kaggle:

| API / Pattern | Code Locations | Original Context | Modern Kaggle Status | Severity Classification |
| :--- | :--- | :--- | :--- | :--- |
| `torchvision._internally_replaced_utils.load_state_dict_from_url` | `models/resnet.py:7`<br>`models/resnet_decoder.py:4` | Internal helper used to download ResNet-50 weights | Removed in `torchvision>=0.14`. Raises `ModuleNotFoundError`. | **MUST PATCH** |
| `torch.cuda.amp.autocast`, `GradScaler` | `methods/edc1.py:10, 69–70` | Mixed precision context and gradient scaler | Supported in PyTorch 2.x (alias to `torch.amp`). Also, `--amp False` by default. | **SAFE** |
| `.data` parameter manipulation | `train_utils.py:419–444, 459–468` | Used inside unused `EMA` and `Bn_Controller` classes | Supported in PyTorch 2.x; these routines are not invoked during training. | **SAFE** |
| `torch.autograd.Variable` | *None* (0 occurrences) | Deprecated autograd wrapper | Completely absent from codebase. Clean tensor usage. | **SAFE** |
| `F.sigmoid` / `F.softmax` | *None* (`models/edc.py:22` uses `nn.Softmax(dim=-1)`) | Functional activations | Modern module-based activation is clean and compliant. | **SAFE** |
| `F.interpolate` / `F.normalize` | `models/edc.py:169–173, 182`<br>`methods/edc1.py:256` | Anomaly map upsampling and standard deviation | Fully backward- and forward-compatible. | **SAFE** |
| `torch.load` | `methods/edc1.py:297` | Checkpoint resuming | Fully compatible. (Kaggle default uses `weights_only=False`). | **SAFE** |
| `LambdaLR` scheduler | `train_utils.py:4, 303, 340` | Learning rate multiplier schedule | Fully compatible across PyTorch 1.x and 2.x. | **SAFE** |
| `torch.optim.AdamW` | `train_utils.py:220, 261` | Decoupled weight decay optimizer | Standard core PyTorch API. | **SAFE** |

---

## 2. GPU & CUDA Dependency Audit

All CUDA references across the repository were catalogued:

| File | Line Number(s) | Code Snippet | Compatibility Assessment |
| :--- | :--- | :--- | :--- |
| `e2ad_br35h.py` | 34–35 | `torch.cuda.manual_seed(seed)`<br>`torch.cuda.manual_seed_all(seed)` | Standard CUDA seed initialization; **SAFE**. |
| `e2ad_br35h.py` | 102–103 | `if not torch.cuda.is_available():`<br>`    raise Exception('ONLY GPU TRAINING IS SUPPORTED')` | Guardrail enforcing CUDA; passes on Kaggle GPU. |
| `e2ad_br35h.py` | 105–106 | `torch.cuda.set_device(args.gpu)`<br>`runner.model = runner.model.cuda(args.gpu)` | Sets device context. **WARNING**: Script default `--gpu '1'` fails on single-GPU sessions. Must pass `--gpu 0`. |
| `e2ad_br35h.py` | 219 | `parser.add_argument('--gpu', default='1', type=str)` | Hardcoded default to device 1. |
| `methods/edc1.py` | 61–64 | `start_batch = torch.cuda.Event(enable_timing=True)`<br>`start_run = torch.cuda.Event(...)` | Modern CUDA event profiling; fully supported. |
| `methods/edc1.py` | 87, 115 | `torch.cuda.synchronize()` | Standard stream barrier; fully supported. |
| `methods/edc1.py` | 90, 188 | `x = x.cuda(args.gpu)`<br>`x, y = x.cuda(args.gpu), y.cuda(args.gpu).float()` | Explicit device placement matching model device; **SAFE**. |
| `datasets/dataset.py` | 155 | `return img_n.cuda(self.gpu), target` | Only reached if `swa_epoch != 0`. Unreachable in BR35H (`swa_epoch=0`). |

**Conclusion**:
- GPU index `0` works flawlessly.
- Model and data tensors reside on the exact same device (`cuda:args.gpu`).
- No multi-device mismatch errors exist as long as `--gpu 0` is supplied.

---

## 3. Data Loader Audit

- **Training Dataset**: Contains exclusively `train/NORMAL/` images (1,000 healthy brain MRIs). No abnormal images are loaded.
- **Evaluation Dataset**: Contains `test/NORMAL/` (500 healthy brain MRIs, target `0.0`) and `test/ABNORMAL/` (1,500 tumor brain MRIs, target `1.0`).
- **Batch Sizes**:
  - Training batch size: `32`
  - Evaluation batch size: `64`
- **Sampling & Workers**:
  - `RandomSampler(replacement=True, num_samples=128000)`: Draws batches from the 1,000 normal images for 4,000 iterations.
  - `num_workers = 4` (Kaggle environments provide 2–4 vCPUs; 4 workers run stably).
  - `pin_memory = False` (default).
  - `drop_last = True` for training batch sampler; `drop_last = False` for evaluation loader.
- **Filesystem Mount Support**:
  - Kaggle mounts datasets under `/kaggle/input/<dataset-slug>/`.
  - Passing `--data_dir /kaggle/input/br35h/BR35H/` maps directly:
    - `/kaggle/input/br35h/BR35H/train/NORMAL` $\rightarrow$ Valid
    - `/kaggle/input/br35h/BR35H/test/NORMAL` $\rightarrow$ Valid
    - `/kaggle/input/br35h/BR35H/test/ABNORMAL` $\rightarrow$ Valid
  - **No path concatenation modifications are required.**

---

## 4. Image & Transform Pipeline Audit

Exact pipeline applied to every input image prior to entering ResNet-50:

1. **Reading**: Loaded via PIL `Image.open(path).convert('RGB')` (ensuring 3 channels).
2. **Spatial Resizing**: Albumentations `A.Resize(256, 256)` (bilinear interpolation).
3. **Cropping**: `A.CenterCrop(256, 256)` (identity crop; maintains $256 \times 256$).
4. **Normalization**: `A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0)`.
5. **Tensor Conversion**: `ToTensorV2()` converts $(H, W, C)$ uint8/float to PyTorch FloatTensor $(C, H, W)$.
6. **Data Augmentation**: None (disabled by `transform=False`).
7. **TCCL Rotational View**: `x_rot = torch.rot90(x, 2, (2, 3))` applies a $180^\circ$ spatial rotation on GPU VRAM.
8. **Final Tensor Spec Entering ResNet-50**:
   $$\text{Tensor Dtype: } \text{torch.float32}, \quad \text{Shape: } [32, 3, 256, 256]$$

---

## 5. Model Weight Audit

- **Backbone**: Modified ResNet-50 (`models.resnet.resnet50(pretrained=True)`).
- **Download URL**: `https://download.pytorch.org/models/resnet50-0676ba61.pth` (ImageNet-1k, ~98 MB).
- **Mechanism**: Fetched by `load_state_dict_from_url` into PyTorch hub cache directory (`~/.cache/torch/hub/checkpoints/`).
- **Kaggle Dependency**:
  - Kaggle notebook **Internet toggle** must be enabled for automatic download on the first run.
  - *Offline Alternative*: The `.pth` file can be attached as a Kaggle dataset and pre-copied to `/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`.
- **Local Checkpoints**: The cloned repository contains **zero** pretrained weights; weights are initialized via ImageNet backbone download.

---

## 6. Checkpoint & Output Audit

- **Output Directory**:
  ```python
  save_path = './{}/{}/{}/{}/'.format(args.save_dir, args.save_name, args.model_name, i)
  ```
  Default: `./saved_models/e2ad_br35h/E2AD/0/` (relative to current working directory).
- **Kaggle Working Directory**: Executing from `/kaggle/working/E2AD/` places saved weights at `/kaggle/working/E2AD/saved_models/e2ad_br35h/E2AD/0/`, ensuring they appear in Kaggle output artifacts.
- **`save_weight` Flag**:
  - Default: `0` (**checkpoints disabled**).
  - `--save_weight 1` **MUST** be passed to save `best_auc.pth` and `last_epoch.pth`.
- **Evaluation**: Fully self-contained. The training script automatically evaluates the test set every 100 iterations and outputs final AUROC, F1, Accuracy, Sensitivity, and Specificity. No external evaluation script is required.

---

## 7. Multi-Run & Seed Audit

- **Default Seed**: `0` (`--seed 0`).
- **Deterministic Flags**:
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - `random.seed(0)`, `np.random.seed(0)`, `torch.manual_seed(0)`, `generator_lb.manual_seed(0)`
- **`train_times` Loop Issue**:
  - Inside `for i in range(args.train_times):`, `args.seed` is never updated.
  - Running `--train_times 5` will execute the exact same deterministic seed 0 calculation five times in succession.
  - To test variance across random initializations on Kaggle, the seed must be varied per run ($seed = args.seed + i$).

---

## 8. Requirements Comparison

| Package | Original Version | Modern Kaggle Version | Risk Level | Action / Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **`torch`** | `1.12.0+cu113` | `2.1.2` to `2.4.0+cu121` | **HIGH** if installing old; **NONE** if using native | Use Kaggle pre-installed PyTorch. |
| **`torchvision`** | `0.13.0+cu113` | `0.16.2` to `0.19.0+cu121` | **CRITICAL** (deleted internal module) | Apply backward-compatible patch in `resnet.py` and `resnet_decoder.py`. |
| **`albumentations`** | *Missing* | `1.4.0`+ | **CRITICAL** if missing | Pre-installed on Kaggle; verify with import. |
| **`pyyaml`** | *Missing* | `6.0`+ | **MEDIUM** if missing | Pre-installed on Kaggle. |
| **`numpy`** | `1.18.4` | `1.24.3` to `1.26.4` | **HIGH** if installing old | Use Kaggle pre-installed NumPy. |
| **`pandas`** | `1.3.5` | `2.0.0`+ | **HIGH** if installing old | Use Kaggle pre-installed Pandas. |
| **`scikit-learn`** | `0.22.2.post1` | `1.2.2`+ | **HIGH** if installing old | Use Kaggle pre-installed Scikit-learn (metrics APIs are identical). |
| **`opencv-python`**| `4.6.0.66` | `4.8.0`+ | **LOW** | Pre-installed on Kaggle. |
| **`Pillow`** | `9.0.1` | `10.0.0`+ | **LOW** | Pre-installed on Kaggle. |
| **`tensorboard`** | `2.11.0` | `2.15.0`+ | **LOW** | Pre-installed on Kaggle. |
| **`tqdm`** | `4.64.1` | `4.66.0`+ | **LOW** | Pre-installed on Kaggle. |

---

## 9. Exact Kaggle Execution Plan

Assuming preprocessed `BR35H/` is mounted at `/kaggle/input/br35h/BR35H/`:

### Step 1: Clone Repository
```bash
git clone https://github.com/TumCCC/E2AD.git /kaggle/working/E2AD
cd /kaggle/working/E2AD
```

### Step 2: Verify Kaggle Environment & GPU
```bash
python -c "import torch, torchvision, albumentations, sklearn; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Step 3: Apply Minimal Compatibility Patch
Update the two lines importing `load_state_dict_from_url` in `models/resnet.py` and `models/resnet_decoder.py`:
```bash
python -c "
for path in ['models/resnet.py', 'models/resnet_decoder.py']:
    with open(path, 'r') as f:
        content = f.read()
    content = content.replace(
        'from torchvision._internally_replaced_utils import load_state_dict_from_url',
        'try:\n    from torch.hub import load_state_dict_from_url\nexcept ImportError:\n    from torchvision._internally_replaced_utils import load_state_dict_from_url'
    )
    with open(path, 'w') as f:
        f.write(content)
print('Compatibility patch applied successfully.')
"
```

### Step 4: Verify Dataset Structure
```bash
python -c "
import os
base = '/kaggle/input/br35h/BR35H'
print('Train NORMAL count:', len(os.listdir(f'{base}/train/NORMAL')))
print('Test NORMAL count :', len(os.listdir(f'{base}/test/NORMAL')))
print('Test ABNORMAL count:', len(os.listdir(f'{base}/test/ABNORMAL')))
"
```

### Step 5: Official BR35H Baseline Training & Evaluation
```bash
python e2ad_br35h.py \
  --train_times 1 \
  --gpu 0 \
  --model_name E2AD \
  --data_dir /kaggle/input/br35h/BR35H/ \
  --save_weight 1
```

### Expected Output Locations:
- Best Model Checkpoint: `/kaggle/working/E2AD/saved_models/e2ad_br35h/E2AD/0/best_auc.pth`
- Final Model Checkpoint: `/kaggle/working/E2AD/saved_models/e2ad_br35h/E2AD/0/last_epoch.pth`
- TensorBoard Event Logs: `/kaggle/working/E2AD/saved_models/e2ad_br35h/E2AD/0/tb/`
- Evaluation Metrics: Logged to console every 100 iterations and summarized at iteration 4,000.

---

## 10. Reproducibility Classification

**Classification**: **`B. READY AFTER MINOR PATCHES`**

### Summary of Distinctions:
1. **OFFICIAL CODE (Untouched)**:
   - Architecture: ResNet-50 encoder + 4 Spatial Attention modules + dual ResNet-50 decoders with skip connections.
   - Loss formulation: $\mathcal{L} = 0.5 \cdot \mathcal{L}_O + 0.5 \cdot \mathcal{L}_T + 0.5 \cdot \mathcal{L}_{\text{TCCL}}$.
   - Training parameters: AdamW ($lr_{\text{enc}} = 5 \times 10^{-5}$, $lr_{\text{dec}} = 5 \times 10^{-4}$), batch size 32, 4,000 iterations.
   - Anomaly score aggregation: Spatial `max` reduction averaged across both branches.
2. **COMPATIBILITY PATCH (Mandatory for modern Torchvision)**:
   - Fallback import for `load_state_dict_from_url` via `torch.hub`.
   - Passing `--gpu 0` to target the active Kaggle accelerator.
3. **OPTIONAL EXPERIMENTAL CHANGE**:
   - Incrementing `args.seed = args.seed + i` if running multi-seed `--train_times 5`.
