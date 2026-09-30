# E2AD BR35H Execution & Data-Flow Trace

**Date**: September 29, 2026  
**Subject**: Complete code-level execution and data-flow trace for BR35H on E2AD  
**Files Audited**:
- `datasets/dataset.py`
- `datasets/data_utils.py`
- `datasets/transforms.py`
- `models/edc.py`
- `models/resnet.py`
- `models/resnet_decoder.py`
- `methods/edc1.py`
- `e2ad_br35h.py`
- `train_utils.py`
- `utils.py`

---

## 1. Dataset Loading

### Class & Instantiation
In `e2ad_br35h.py` (lines 44–50), the dataset is instantiated twice using `AD_Dataset` from `datasets/dataset.py`:
```python
# Construct Dataset & DataLoader
train_dset = AD_Dataset(name=args.dataset, train=True, transform=False, data_dir=args.data_dir)
train_dset = train_dset.get_dset()

eval_dset = AD_Dataset(name=args.dataset, train=False, transform=False, data_dir=args.data_dir)
eval_dset = eval_dset.get_dset()
```
The inner dataset returned by `.get_dset()` is an instance of `BasicDataset` (subclass of `torch.utils.data.Dataset`).

### Directory Structure & Path Expectations
In `datasets/dataset.py` (lines 206–225 in `AD_Dataset.get_data`):
- **Training set (`train=True`)**:
  ```python
  train_path = os.path.join(self.data_dir, 'train', 'NORMAL')
  norm_files = os.listdir(train_path)
  ```
  It **strictly expects** `{data_dir}/train/NORMAL`. It does not look for or touch any other subfolder in `train`.
- **Evaluation set (`train=False`)**:
  ```python
  for sub_dir in os.listdir(os.path.join(self.data_dir, 'test')):
      files = os.listdir(os.path.join(self.data_dir, 'test', sub_dir))
      paths = [os.path.join(self.data_dir, 'test', sub_dir, file) for file in files]
      img_paths.extend(paths)
      if sub_dir == 'NORMAL':
          targets.extend(list(np.zeros(len(paths))))
      else:
          targets.extend(list(np.ones(len(paths))))
  ```
  It iterates over all subdirectories in `{data_dir}/test`. Any folder named `'NORMAL'` is assigned target `0.0`. Any other folder (such as `'ABNORMAL'`) is assigned target `1.0`.

### Labels
- **NORMAL**: `0.0`
- **ABNORMAL**: `1.0`

### Loader Contents & Sampling
- **Does training loader ever see ABNORMAL images?**
  **NO.** `train_dset.get_data()` only enumerates `train/NORMAL/`. The 1,500 tumor images in `test/ABNORMAL/` are completely invisible during training.
- **Does test loader contain both normal and abnormal images?**
  **YES.** It loads all 500 images from `test/NORMAL/` (target 0.0) and all 1,500 images from `test/ABNORMAL/` (target 1.0), totaling 2,000 evaluation images.
- **Hidden filtering or sampling**:
  - `AD_Dataset.get_data()` has a ceiling `self.train_samples_limit = 30000` (line 209). Since BR35H has 1,000 training images, no sampling limit triggers.
  - The training DataLoader uses `RandomSampler` with `replacement=True` and `num_samples = batch_size * num_iters = 32 * 4000 = 128,000` (`datasets/data_utils.py` line 127). The 1,000 normal images are randomly sampled with replacement across 4,000 iteration batches.

---

## 2. Transform Pipeline

### Code Tracing in `datasets/dataset.py`

1. **Image Loading**:
   In `BasicDataset.__getitem__` (line 146):
   ```python
   img = default_loader(self.img_paths[idx])
   img = np.array(img)
   ```
   `default_loader` invokes `pil_loader` (line 43):
   ```python
   with open(path, 'rb') as f:
       img = Image.open(f)
       return img.convert('RGB')
   ```
   Images are always loaded as 3-channel RGB.

2. **Spatial Resizing & Cropping**:
   In `e2ad_br35h.py`, `transform=False` is passed to `AD_Dataset`, which selects `get_transform(img_size, crop_size, train)` (lines 60–75):
   ```python
   transform = A.Compose([
       A.Resize(img_size, img_size),
       A.CenterCrop(crop_size, crop_size),
   ])
   ```
   For BR35H, `args.img_size = 256` and default `crop_size = 256`. Therefore:
   - `A.Resize(256, 256)` resizes any arbitrary input dimension (e.g. `630x630` or `348x287`) to $256 \times 256$.
   - `A.CenterCrop(256, 256)` on a $256 \times 256$ image is an identity crop.

3. **Data Augmentation**:
   **NONE.** No random horizontal flip, vertical flip, random rotation, color jitter, or noise is applied. Augmentation (`A.RandomRotate90()`) exists only in `get_transform_tta`, which is disabled (`transform=False`).

4. **Normalization & Tensor Conversion**:
   In `BasicDataset.__init__` (line 126):
   ```python
   self.totensor = A.Compose([
       A.Normalize() if imagenet_norm else A.Lambda(image=divide255),
       ToTensorV2()
   ])
   ```
   - `A.Normalize()` uses default ImageNet statistics:
     $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
   - `ToTensorV2()` permutes dimensions from $(H, W, C)$ to PyTorch Tensor $(C, H, W)$ float32.

5. **TCCL Rotational Transformation**:
   In `models/edc.py` (line 118 in `E2AD.forward`):
   ```python
   x_rot = torch.rot90(x, 2, (2, 3))
   ```
   A deterministic $180^\circ$ spatial rotation is applied directly on the normalized GPU tensor batch across height (axis 2) and width (axis 3).

### Exact Tensor Dimension Entering ResNet-50
$$x \in \mathbb{R}^{B \times 3 \times 256 \times 256} \quad (\text{with } B=32 \text{ in training, } B=64 \text{ in eval})$$

---

## 3. BR35H Configuration

| Parameter | Value in `e2ad_br35h.py` | Code Location |
| :--- | :--- | :--- |
| **Model Name** | `'E2AD'` | line 198 |
| **Input Image Size / Crop Size** | $256 \times 256$ / $256 \times 256$ | line 193 / `dataset.py:171` |
| **Training Batch Size** | `32` | line 171 |
| **Evaluation Batch Size** | `64` | line 172 |
| **Number of Training Iterations** | `4,000` | line 167 |
| **Evaluation Frequency** | Every `100` iterations | line 169 |
| **Epoch Argument** | `1` (Unused; loop bounded by iterations) | line 166 |
| **Optimizer** | `AdamW` | line 179 |
| **Decoder / Attention LR** | `5e-4` | line 180 |
| **Encoder LR** | `5e-5` | line 181 |
| **Weight Decay** | `1e-4` | line 183 |
| **Gradient Clipping (`clip`)** | `1.0` | line 185 |
| **LR Scheduler** | Warmup MultiStep (`milestones=[1e10]`, `gamma=0.2`, `warmup=0`) $\rightarrow$ **Constant LR** | lines 98–99 |
| **Encoder BatchNorm Mode** | `bn_pretrain=False` (Encoder BN tracked/trained) | line 71 |
| **BatchNorm Momentum** | `0.01` (overrides PyTorch default 0.1) | lines 88–90 |
| **Anomaly Score Aggregation** | `'max'` (default in `EDC_MS`) | `methods/edc1.py:20` |
| **Threshold Selection Method** | Max F1 score on Precision-Recall curve (`return_best_thr`) | `methods/edc1.py:264, 582` |
| **TCCL / Loss Coefficients** | $\lambda_O = 0.5, \ \lambda_T = 0.5, \ \lambda_{\text{TCCL}} = 0.5$ | `models/edc.py:167` |
| **Checkpoint Saving (`save_weight`)**| `0` (Disabled by default) | line 199 |
| **Default GPU ID** | `'1'` | line 219 |
| **Random Seed** | `0` | line 217 |

---

## 4. Model Architecture & Tensor Flow

```
BR35H Image (H, W, 3)
   │
   ▼ [Resize(256, 256) + Normalize + ToTensorV2]
x : [B, 3, 256, 256] ────────────────────────────┐
   │                                             │ [rot90(x, 2, (2,3))]
   ▼                                             ▼
ResNet-50 Encoder                             ResNet-50 Encoder
   ├─► e1_1 : [B, 256, 64, 64]                   ├─► e1_2 : [B, 256, 64, 64]
   ├─► e2_1 : [B, 512, 32, 32]                   ├─► e2_2 : [B, 512, 32, 32]
   ├─► e3_1 : [B, 1024, 16, 16]                  ├─► e3_2 : [B, 1024, 16, 16]
   └─► e4_1 : [B, 2048, 8, 8]                    └─► e4_2 : [B, 2048, 8, 8]
        │         │                                   │         │
        ▼         ▼                                   ▼         ▼
    avgpool    sa2(e2_1) & sa1(e3_1)              avgpool    sa4(e2_2) & sa3(e3_2)
        │         │                                   │         │
        ▼         ▼                                   ▼         ▼
  e4_1_global  e2_1_sa [B, 512, 32, 32]         e4_2_global  e2_2_sa [B, 512, 32, 32]
   [B, 2048]   e3_1_sa [B, 1024, 16, 16]         [B, 2048]   e3_2_sa [B, 1024, 16, 16]
        │         │                                   │         │
        │         ▼                                   │         ▼
        │     Decoder 1 (ResNetDecoder_SC)            │     Decoder 2 (ResNetDecoder_SC)
        │     layer3(e4_1)                            │     layer3(e4_2)
        │     layer2(f3 + e3_1_sa)                    │     layer2(f3 + e3_2_sa)
        │     layer1(f2 + e2_1_sa)                    │     layer1(f2 + e2_2_sa)
        │         ├─► d1_1 : [B, 256, 64, 64]         │         ├─► d1_2 : [B, 256, 64, 64]
        │         ├─► d2_1 : [B, 512, 32, 32]         │         ├─► d2_2 : [B, 512, 32, 32]
        │         └─► d3_1 : [B, 1024, 16, 16]        │         └─► d3_2 : [B, 1024, 16, 16]
        │                                             │
        ▼                                             ▼
   TCCL Loss: 1 - cosine_sim(e4_1_global, e4_2_global)
```

### Anomaly Map and Image-Level Score Calculation
During evaluation (`methods/edc1.py` lines 192–239):
1. **Pixel-level cosine distance**:
   $$p1 = 1 - \text{cosine\_sim}(d1_1, e1_1) \in [B, 1, 64, 64]$$
   $$p2 = 1 - \text{cosine\_sim}(d2_1, e2_1) \in [B, 1, 32, 32] \xrightarrow{\text{interp } \times 2} [B, 1, 64, 64]$$
   $$p3 = 1 - \text{cosine\_sim}(d3_1, e3_1) \in [B, 1, 16, 16] \xrightarrow{\text{interp } \times 4} [B, 1, 64, 64]$$
2. **Multi-scale feature fusion**:
   $$p\_all\_1 = \text{mean}(p1, p2, p3) \in [B, 1, 64, 64]$$
   $$p\_all\_2 = \text{mean}(p4, p5, p6) \in [B, 1, 64, 64]$$
3. **Spatial Aggregation (`max` reduction)**:
   $$p\_img\_1 = \max_{(h,w)} p\_all\_1 \in [B]$$
   $$p\_img\_2 = \max_{(h,w)} p\_all\_2 \in [B]$$
   $$\text{Anomaly Score} = \frac{p\_img\_1 + p\_img\_2}{2} \in [B]$$

---

## 5. Loss Formulation

In `models/edc.py` (lines 145–167):

1. **Original View Reconstruction Loss ($L_O$)**:
   $$l_1 = 1 - \text{cosine\_sim}(d1_1^{\text{flat}}, e1_1^{\text{flat}})$$
   $$l_2 = 1 - \text{cosine\_sim}(d2_1^{\text{flat}}, e2_1^{\text{flat}})$$
   $$l_3 = 1 - \text{cosine\_sim}(d3_1^{\text{flat}}, e3_1^{\text{flat}})$$
   $$L_O = l_1 + l_2 + l_3$$

2. **Transformed View Reconstruction Loss ($L_T$)**:
   $$l_4 = 1 - \text{cosine\_sim}(d1_2^{\text{flat}}, e1_2^{\text{flat}})$$
   $$l_5 = 1 - \text{cosine\_sim}(d2_2^{\text{flat}}, e2_2^{\text{flat}})$$
   $$l_6 = 1 - \text{cosine\_sim}(d3_2^{\text{flat}}, e3_2^{\text{flat}})$$
   $$L_T = l_4 + l_5 + l_6$$

3. **Transformation Consistency Contrastive Loss ($L_{\text{TCCL}}$)**:
   $$L_{\text{TCCL}} = 1 - \text{cosine\_sim}(e4_1^{\text{global}}, e4_2^{\text{global}})$$

4. **Total Training Loss ($L_{\text{final}}$)**:
   $$L_{\text{final}} = (0.5 \cdot L_O + 0.5 \cdot L_T) + 0.5 \cdot L_{\text{TCCL}}$$

---

## 6. GPU / CUDA Dependencies

Every GPU-dependent call is catalogued below:

### In `e2ad_br35h.py`:
- **Line 34–35**: `torch.cuda.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`
- **Line 102–103**:
  ```python
  if not torch.cuda.is_available():
      raise Exception('ONLY GPU TRAINING IS SUPPORTED')
  ```
- **Line 105–106**: `torch.cuda.set_device(args.gpu)`, `runner.model = runner.model.cuda(args.gpu)`
- **Line 219**: Default argument is `--gpu '1'` (must be overridden to `0` for single-GPU environments).

### In `methods/edc1.py`:
- **Line 10**: `from torch.cuda.amp import autocast, GradScaler`
- **Lines 61–64**:
  ```python
  start_batch = torch.cuda.Event(enable_timing=True)
  end_batch = torch.cuda.Event(enable_timing=True)
  start_run = torch.cuda.Event(enable_timing=True)
  end_run = torch.cuda.Event(enable_timing=True)
  ```
- **Lines 87–88, 114–115**: `torch.cuda.synchronize()` on every iteration.
- **Lines 90, 188**: `x = x.cuda(args.gpu)`, `x, y = x.cuda(args.gpu), y.cuda(args.gpu).float()`

---

## 7. Checkpoint Behavior

1. **Default State**:
   - `args.save_weight` defaults to `0` (`e2ad_br35h.py:199`).
   - Checkpoints are **NOT** saved unless `--save_weight 1` is explicitly provided.
2. **Saved Files** (when `save_weight=1`):
   - `best_auc.pth`: Saved at `save_path/best_auc.pth` whenever evaluation AUROC reaches a new maximum (`methods/edc1.py:138`).
   - `last_epoch.pth`: Saved at `save_path/last_epoch.pth` at the end of iteration 4,000 (`methods/edc1.py:154`).
3. **Directory Path**:
   `save_path = './{args.save_dir}/{args.save_name}/{args.model_name}/{i}/'`  
   Default: `./saved_models/e2ad_br35h/E2AD/0/`
4. **Resume Support**:
   - Supported via `--resume 1 --load_path <path>`.
   - `runner.load_model` expects a dict: `{'model', 'optimizer', 'scheduler', 'it'}`. (Note: `best_auc.pth` saves only `state_dict()`, whereas `save_model()` saves the dictionary).
5. **Backbone Weights**:
   - ResNet-50 ImageNet-1k weights (`resnet50-0676ba61.pth`, ~98 MB) are automatically fetched by `load_state_dict_from_url` into PyTorch hub cache on the first run.

---

## 8. Random Seed & Multi-Run Behavior

In `e2ad_br35h.py`:
```python
def main_worker(gpu, args, total_list, save_path):
    args.gpu = gpu
    seed = args.seed 
    cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```
And the outer execution loop (lines 225–243):
```python
total_list = [[], [], [], []]
for i in range(args.train_times):
    save_path = './{}/{}/{}/{}/'.format(args.save_dir, args.save_name, args.model_name, i)
    total_list = main_worker(int(args.gpu), args, total_list, save_path)
```

> [!WARNING]
> **Identical Multi-Run Bug Discovered**:
> Inside `for i in range(args.train_times):`, `args.seed` is **never updated** or varied by `i`.
> In each run ($i = 0, 1, 2, 3, 4$), `seed = args.seed = 0` is set with `cudnn.deterministic = True`.
> Consequently, running `--train_times 5` without incrementing the seed in code or via config files will re-execute the exact same deterministic computation 5 times, yielding identical results across all 5 runs.
> For true multi-seed experiments on Kaggle, the seed must be varied per run ($seed = args.seed + i$).

---

## 9. Compatibility Issues for Kaggle GPU

1. **`torchvision._internally_replaced_utils` removal**:
   - In `models/resnet.py:7` and `models/resnet_decoder.py:4`:
     ```python
     from torchvision._internally_replaced_utils import load_state_dict_from_url
     ```
   - In Kaggle's environment (torchvision $\ge 0.14$), this internal module does not exist and throws `ModuleNotFoundError`.
   - Patch: Use `from torch.hub import load_state_dict_from_url`.
2. **Default GPU device ID**:
   - The CLI argument defaults to `--gpu '1'`. On standard Kaggle single-GPU notebook sessions (`CUDA_VISIBLE_DEVICES=0`), running without `--gpu 0` crashes with `CUDA error: invalid device ordinal`.
   - Always run with `--gpu 0`.
3. **Missing `pyyaml` / `albumentations` in requirements**:
   - Both are pre-installed in Kaggle's standard container, but must be kept in mind if building custom environments.
