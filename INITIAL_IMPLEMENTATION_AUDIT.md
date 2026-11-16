# INITIAL IMPLEMENTATION AUDIT REPORT

**Project**: E2AD + R-CAR: Rotation-Consistent Anomaly Reconstruction for Medical Image Anomaly Detection  
**Target Benchmark**: OCT2017 (Retinal Optical Coherence Tomography)  
**Reference Paper**: *"Anomaly Detection in Medical Images Using Encoder-Attention-2Decoders Reconstruction"*, IEEE Transactions on Medical Imaging (TMI), Vol. 44, No. 8, August 2025 (Tang et al.)  
**Official Reference Repository**: `https://github.com/TumCCC/E2AD`  
**Researcher Repository**: `https://github.com/gnanadeep256/E2AD`  
**Target Environment**: Kaggle (Tesla T4 GPU, ~14.5 GB VRAM)  
**Audit Date**: November 2026  

---

## 1. Executive Summary

This forensic audit analyzes the existing E2AD reproduction and preliminary R-CAR implementation (`notebook7cca833b44.ipynb`, `models/edc.py`, `methods/edc1.py`, `datasets/dataset.py`, `e2ad_oct.py`, and `baseline notebooks/oct2017.ipynb`).

While genuine OCT2017 dataset discovery and basic model structures are in place, several critical methodological and structural issues were identified:
1. **Duplicate Forward Pass in Subclass**: `E2AD_RCAR_Model` invokes `super().forward(x)` and then re-runs the entire encoder, attention, and dual decoders. This doubles forward compute, doubles VRAM usage, causes redundant and discordant BatchNorm running statistics updates, and computes the E2AD loss and R-CAR loss from separate activations.
2. **Scoring Function Discrepancy (`max` vs `mean`)**: The published paper (Section IV-B) and reference codebase (`methods/edc1.py:EDC_MS(amap_reduction='max')`) specify **maximum reduction** across spatial dimensions for OCT2017 anomaly scoring. The existing R-CAR notebook used **mean reduction**, causing the apparent baseline performance on held-out test sets to drop from ~99.8% to ~85.3%.
3. **Uncontrolled Random Initializations Across Arms**: In the current notebook, each experiment arm instantiates a fresh `E2AD_RCAR_Model(bn_pretrain=True)` with random decoder weights rather than loading a shared, frozen `initial_model_state.pth`.
4. **Learning Rate Scheduler Milestones**: The scheduler is configured with `milestones=[100000]`, which is never reached during 2,000 iterations. In the author's reference codebase (`e2ad_oct.py`), `milestones=[1e10]` was used, effectively keeping the learning rate constant throughout training.
5. **Partial Dataset Verification**: Only a subset of training images was verified for integrity in earlier passes.

---

## 2. Existing Architecture & Forward-Pass Flow

### 2.1 Model Components
The E2AD architecture consists of four distinct sub-networks:
1. **Encoder ($E$)**: ResNet-50 backbone initialized with ImageNet pre-trained weights.
   - Extracts multi-scale feature hierarchies:
     - Stage 1 ($e_1$): shape $[B, 256, 64, 64]$
     - Stage 2 ($e_2$): shape $[B, 512, 32, 32]$
     - Stage 3 ($e_3$): shape $[B, 1024, 16, 16]$
     - Stage 4 ($e_4$): shape $[B, 2048, 8, 8]$
   - Global pooled representation: $e_{4, \text{global}} = \text{avgpool}(e_4) \in [B, 2048]$
2. **Self-Attention Skip Connections (SASC)**:
   - Position Attention Modules (PAM) applied to intermediate encoder features:
     - $\text{SA}_2$ on $e_2 \to e_{2, \text{sa}} \in [B, 512, 32, 32]$
     - $\text{SA}_1$ on $e_3 \to e_{3, \text{sa}} \in [B, 1024, 16, 16]$
   - For rotated input $X_{\text{rot}}$: $\text{SA}_4$ and $\text{SA}_3$ are used identically.
3. **Decoder 1 ($D_1$)**: ResNet-50 decoder with skip connections, reconstructing features of the original image $X$:
   - Takes $[e_1, e_{2, \text{sa}}, e_{3, \text{sa}}, e_4]$
   - Outputs reconstructed multi-scale features:
     - $d_1^1 \in [B, 256, 64, 64]$ (matches $e_1$)
     - $d_2^1 \in [B, 512, 32, 32]$ (matches $e_2$)
     - $d_3^1 \in [B, 1024, 16, 16]$ (matches $e_3$)
4. **Decoder 2 ($D_2$)**: Independent ResNet-50 decoder with skip connections, reconstructing features of the $180^\circ$ rotated image $X_{\text{rot}}$:
   - Takes $[e_1^{\text{rot}}, e_{2, \text{sa}}^{\text{rot}}, e_{3, \text{sa}}^{\text{rot}}, e_4^{\text{rot}}]$
   - Outputs reconstructed multi-scale features:
     - $d_1^2 \in [B, 256, 64, 64]$ (matches $e_1^{\text{rot}}$)
     - $d_2^2 \in [B, 512, 32, 32]$ (matches $e_2^{\text{rot}}$)
     - $d_3^2 \in [B, 1024, 16, 16]$ (matches $e_3^{\text{rot}}$)

### 2.2 Forward-Pass Flow Diagram
```text
Input X [B, 3, 256, 256] ────────┬──> Encoder(X) ────────> e1, e2, e3, e4 ──> SASC ──> Decoder 1 ──> d1_1, d2_1, d3_1
                                  │
X_rot = rot90(X, 2) [B, 3, 256, 256] ─┴──> Encoder(X_rot) ────> e1_r, e2_r, e3_r, e4_r ──> SASC ──> Decoder 2 ──> d1_2, d2_2, d3_2
```

---

## 3. Original E2AD Loss Components

The total training loss in the reference paper (Eq. 12) is:
$$\mathcal{L}_{\text{E2AD}} = \frac{1}{2} \mathcal{L}_O + \frac{1}{2} \mathcal{L}_T + \alpha_{\text{aux}} \mathcal{L}_{\text{aux}}$$
where $\alpha_{\text{aux}} = 0.5$ for OCT2017.

### 3.1 Original Reconstruction Loss ($\mathcal{L}_O$)
Reconstructs original image feature representations using Decoder 1:
$$\mathcal{L}_O = \sum_{k=1}^3 \left( 1 - \text{cosine\_similarity}(d_k^1, \text{detach}(e_k^1)) \right)$$
Flattened per image across channels and spatial locations:
$$l_k = 1 - \frac{1}{B} \sum_{b=1}^B \frac{d_k^1(b) \cdot e_k^1(b)}{\|d_k^1(b)\|_2 \|e_k^1(b)\|_2}$$

### 3.2 Transformed Reconstruction Loss ($\mathcal{L}_T$)
Reconstructs rotated image feature representations using Decoder 2:
$$\mathcal{L}_T = \sum_{k=1}^3 \left( 1 - \text{cosine\_similarity}(d_k^2, \text{detach}(e_k^2)) \right)$$

### 3.3 Transformation-Consistency Contrastive Learning (TCCL / $\mathcal{L}_{\text{aux}}$)
Enforces global semantic consistency between the representations of $X$ and $X_{\text{rot}}$:
$$\mathcal{L}_{\text{aux}} = 1 - \frac{1}{B} \sum_{b=1}^B \frac{e_{4, \text{global}}^1(b) \cdot e_{4, \text{global}}^2(b)}{\|e_{4, \text{global}}^1(b)\|_2 \|e_{4, \text{global}}^2(b)\|_2}$$

> [!IMPORTANT]
> **Gradient Detach Semantics**:
> As specified in `models/edc.py` lines 137-143, the target representations $e_1^1, e_2^1, e_3^1$ and $e_1^2, e_2^2, e_3^2$ are strictly **detached** before calculating cosine reconstruction loss. This prevents the encoder from collapsing towards decoder outputs, forcing the decoders to learn the reverse mapping of ImageNet-pretrained representations. Conversely, $e_{4, \text{global}}^1$ and $e_{4, \text{global}}^2$ are not detached, allowing contrastive alignment gradients to optimize the encoder.

---

## 4. Exact Definitions of Anomaly Residual Maps (`p_all_1` and `p_all_2`)

`p_all_1` and `p_all_2` are **feature-level anomaly residual maps**, **NOT** RGB pixel-space reconstruction errors:

1. For each stage $k \in \{1, 2, 3\}$, the cosine distance map is computed along the channel dimension:
   $$p_k^1(u, v) = 1 - \frac{d_k^1(u, v) \cdot e_k^1(u, v)}{\|d_k^1(u, v)\|_2 \|e_k^1(u, v)\|_2} \in [B, 1, H_k, W_k]$$
2. Spatial dimensions across layers:
   - $p_1^1 \in [B, 1, 64, 64]$ (native scale)
   - $p_2^1 \in [B, 1, 32, 32] \xrightarrow{\text{Bilinear Upsample } \times 2} [B, 1, 64, 64]$
   - $p_3^1 \in [B, 1, 16, 16] \xrightarrow{\text{Bilinear Upsample } \times 4} [B, 1, 64, 64]$
3. Layer-aggregated residual map for Decoder 1:
   $$p_{\text{all}, 1} = \frac{1}{3} \sum_{k=1}^3 p_k^1 \in [B, 1, 64, 64]$$
4. Layer-aggregated residual map for Decoder 2 (in $180^\circ$-rotated coordinate frame):
   $$p_{\text{all}, 2} = \frac{1}{3} \sum_{k=1}^3 p_k^2 \in [B, 1, 64, 64]$$

---

## 5. Preprocessing Specifications

From `datasets/dataset.py` (author commit `d84384f`):
- **Image Loader**: PIL RGB loader (`Image.open(f).convert('RGB')`)
- **Spatial Transform**:
  ```python
  transform = A.Compose([
      A.Resize(256, 256),
      A.CenterCrop(256, 256),
  ])
  ```
- **Tensor Normalization**:
  ```python
  totensor = A.Compose([
      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ToTensorV2()
  ])
  ```
- **Native vs Transformed Aspect Ratio**:
  OCT2017 images have variable native resolutions (e.g. $512 \times 496$, $768 \times 496$, $1024 \times 496$). The author's pipeline applies `A.Resize(256, 256)`, which squashes the aspect ratio directly into $256 \times 256$. `A.CenterCrop(256, 256)` is a no-op because the spatial size is already $256 \times 256$.

---

## 6. Optimizer, Scheduler & Training Configuration

### 6.1 Optimizer Configuration
- **Type**: AdamW (`torch.optim.AdamW`)
- **Parameter Groups**:
  - Group 0 (`edc_encoder`): `lr = 1e-5`, `weight_decay = 1e-4`
  - Group 1 (`sa1, sa2, sa3, sa4`): `lr = 5e-4`, `weight_decay = 1e-4`
  - Group 2 (`edc_decoder_1`): `lr = 5e-4`, `weight_decay = 1e-4`
  - Group 3 (`edc_decoder_2`): `lr = 5e-4`, `weight_decay = 1e-4`
- **Gradient Clipping**: `clip_grad_norm_ = 1.0`
- **Mixed Precision**: Automatic Mixed Precision (`torch.cuda.amp.autocast`)

### 6.2 Scheduler Configuration
- In `e2ad_oct.py`: `get_multistep_schedule_with_warmup(optimizer, milestones=[1e10], gamma=0.2, num_warmup_steps=0)`
- In `notebook7cca833b44.ipynb`: `milestones=[100000]`
- **Effect**: Learning rate remains strictly constant across all 2,000 iterations.

### 6.3 Iterations & Evaluation Frequency
- **Total Training Iterations**: 2,000 iterations
- **Evaluation Interval**: Every 100 iterations (20 evaluation cycles total)
- **Batch Size**: 16 for Kaggle Tesla T4 safety (author default was 32)

---

## 7. Current R-CAR Implementation & Defect Analysis

### 7.1 Mathematical Formulation
The proposed R-CAR regularizer enforces spatial consistency between the feature residual maps of the unrotated and rotated decoders:
1. $A_1 = p_{\text{all}, 1} \in [B, 1, 64, 64]$
2. $A_{2, \text{rot}} = p_{\text{all}, 2} \in [B, 1, 64, 64]$
3. Inverse alignment via $180^\circ$ spatial rotation:
   $$A_2 = \text{rot90}(A_{2, \text{rot}}, 2, \text{dims}=(-2, -1))$$
4. Confidence Weight Map:
   $$w(u, v) = \exp\big(-\alpha \cdot \text{stop\_gradient}(A_1(u, v) + A_2(u, v))\big)$$
5. R-CAR Loss:
   $$\mathcal{L}_{\text{RCAR}} = \frac{\sum_{u, v} w(u, v) \cdot |A_1(u, v) - A_2(u, v)|}{\sum_{u, v} w(u, v) + \epsilon}$$

### 7.2 Defect 1: Duplicate Forward Pass in `E2AD_RCAR_Model`
In `notebook7cca833b44.ipynb` (Cell 10):
```python
class E2AD_RCAR_Model(E2AD):
    def forward(self, x, return_rcar_maps=False):
        res = super(E2AD_RCAR_Model, self).forward(x)  # Forward pass 1
        if getattr(self, 'enable_rcar', False) or return_rcar_maps:
            # Re-runs encoder on x and x_rot, re-runs SA, re-runs decoders 1 and 2!
            _, e1_1, e2_1, e3_1, e4_1 = self.edc_encoder(x)
            ...
            d1_1, d2_1, d3_1 = self.edc_decoder_1(...)
            ...
```
**Consequences**:
- Doubled latency and VRAM consumption.
- BatchNorm modules updated running mean and variance **twice per training step** with different inputs.
- Activations used for $\mathcal{L}_{\text{E2AD}}$ were decoupled from activations used for $\mathcal{L}_{\text{RCAR}}$.

### 7.3 Defect 2: Anomaly Reduction Mode (`max` vs `mean`)
- In `baseline notebooks/oct2017.ipynb`, `amap_reduction = 'max'` yielded:
  - Best AUROC: **99.83%**
  - Final F1: **98.83%**
- In `notebook7cca833b44.ipynb`, `p_img = (p_all_1.mean() + p_all_2.mean()) / 2` was used, yielding **~85.3%**.
- The paper (page 6, section IV-B) confirms: "$\mathcal{M}$ is set to mean mode on ISIC and to maximum mode on the other three datasets [OCT2017, APTOS, Br35H]."

### 7.4 Defect 3: Independent Random Weights Across Arms
- Baseline and R-CAR models were instantiated as separate objects with fresh random seeds, confounding architectural comparison with weight initialization variance.

---

## 8. Artifacts to Preserve

1. `notebook7cca833b44.ipynb` (Working genuine pipeline)
2. `baseline notebooks/oct2017.ipynb` (Canonical baseline execution reference)
3. `models/edc.py` and `methods/edc1.py` (Official author implementation)
4. Checkpoint saving and reload mechanisms (`best_auc.pth`, `last_epoch.pth`)
5. Stratified validation split logic (200 validation images, 800 test images)
6. Qualitative heatmap generation script
