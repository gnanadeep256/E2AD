import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import hashlib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Matplotlib config
os.environ['MPLCONFIGDIR'] = os.path.abspath('scratch/mpl')
os.makedirs('scratch/mpl', exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.test_architecture_equivalence import E2AD_Unified
from tools.test_rcar_validation import RCARLoss

def run_shared_init_and_loss_scale():
    print("=" * 70)
    print("PHASE 6 & 7: SHARED INITIALIZATION, LOSS SCALE & LR SCHEDULE AUDIT")
    print("=" * 70)
    
    device = torch.device('cpu')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Create and Save Shared Initial Model State
    initial_model = E2AD_Unified(bn_pretrain=True).to(device)
    for m in initial_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.01
            
    init_state_path = "initial_model_state.pth"
    torch.save(initial_model.state_dict(), init_state_path)
    
    # Also copy to E2AD_RCAR_OCT2017_CORRECTED
    corr_init_path = "E2AD_RCAR_OCT2017_CORRECTED/initial_model_state.pth"
    torch.save(initial_model.state_dict(), corr_init_path)
    
    # Calculate state dict hash
    hasher = hashlib.sha256()
    with open(init_state_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    init_hash = hasher.hexdigest()
    print(f"Created shared initial model state: {init_state_path}")
    print(f"Initial State SHA-256             : {init_hash}")
    
    # Verify reloading produces exact hash and parameter match
    test_model = E2AD_Unified(bn_pretrain=True).to(device)
    test_model.load_state_dict(torch.load(init_state_path, map_location=device))
    param_diff = sum(torch.abs(p1 - p2).sum().item() for p1, p2 in zip(initial_model.parameters(), test_model.parameters()))
    assert param_diff == 0.0, f"Reloaded model does not match initial state: diff={param_diff}"
    print("Shared initialization reload test : PASS (100% Parameter Match)")
    
    # 2. Loss-Scale Analysis on Sample OCT Batch
    print("\nRunning Loss-Scale and Gradient Norm Ratio Analysis...")
    from PIL import Image
    import torchvision.transforms as T
    
    # Load 4 real OCT normal images
    train_dir = "OCT2017/train/NORMAL"
    sample_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)[:4]]
    
    transform = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    batch_tensors = [transform(Image.open(f).convert('RGB')) for f in sample_files]
    x_batch = torch.stack(batch_tensors, dim=0).to(device)
    
    initial_model.train()
    out = initial_model(x_batch)
    l_e2ad = out['loss']
    A1 = out['A1']
    A2 = out['A2']
    
    rcar_unw_fn = RCARLoss(mode="unweighted")
    l_rcar_unw, m_unw, _ = rcar_unw_fn(A1, A2)
    
    rcar_w_fn = RCARLoss(mode="weighted", alpha=2.0)
    l_rcar_w, m_w, _ = rcar_w_fn(A1, A2)
    
    ratio_unw = l_rcar_unw.item() / (l_e2ad.item() + 1e-8)
    ratio_w = l_rcar_w.item() / (l_e2ad.item() + 1e-8)
    
    print(f"Mean E2AD Loss                   : {l_e2ad.item():.4f}")
    print(f"Mean Unweighted R-CAR Loss       : {l_rcar_unw.item():.4f} (Ratio to E2AD: {ratio_unw:.4f})")
    print(f"Mean Weighted R-CAR Loss         : {l_rcar_w.item():.4f} (Ratio to E2AD: {ratio_w:.4f})")
    
    # Gradient Norm analysis contributed by each objective
    initial_model.zero_grad()
    l_e2ad.backward(retain_graph=True)
    e2ad_dec_grad = initial_model.edc_decoder_1.layer1[0].conv1.weight.grad.norm().item()
    
    initial_model.zero_grad()
    l_rcar_unw.backward(retain_graph=True)
    rcar_unw_dec_grad = initial_model.edc_decoder_1.layer1[0].conv1.weight.grad.norm().item()
    
    initial_model.zero_grad()
    l_rcar_w.backward()
    rcar_w_dec_grad = initial_model.edc_decoder_1.layer1[0].conv1.weight.grad.norm().item()
    
    print(f"Decoder Gradient Norm from E2AD  : {e2ad_dec_grad:.4e}")
    print(f"Decoder Grad Norm from Unw RCAR  : {rcar_unw_dec_grad:.4e} (Ratio: {rcar_unw_dec_grad/e2ad_dec_grad:.4f})")
    print(f"Decoder Grad Norm from W RCAR    : {rcar_w_dec_grad:.4e} (Ratio: {rcar_w_dec_grad/e2ad_dec_grad:.4f})")
    
    # 3. Learning Rate Schedule Figure
    total_iters = 2000
    iters = list(range(1, total_iters + 1))
    enc_lrs = [1e-5] * total_iters
    dec_lrs = [5e-4] * total_iters
    
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(iters, dec_lrs, label="Decoders & SASC LR (5e-4)", color="#1f77b4", linewidth=2)
    ax.plot(iters, enc_lrs, label="Encoder LR (1e-5)", color="#d62728", linewidth=2, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("Training Iteration", fontsize=11)
    ax.set_ylabel("Learning Rate (log scale)", fontsize=11)
    ax.set_title("E2AD Learning Rate Schedule (Milestones=[1e10], Constant LR)", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="center right")
    plt.tight_layout()
    
    fig_path1 = "figures/learning_rate_schedule.png"
    fig_path2 = "E2AD_RCAR_OCT2017_CORRECTED/figures/learning_rate_schedule.png"
    for p in [fig_path1, fig_path2]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved learning rate schedule plot: {fig_path1}")
    
    loss_scale_report = {
        "initial_state_sha256": init_hash,
        "mean_e2ad_loss": float(l_e2ad.item()),
        "mean_unweighted_rcar_loss": float(l_rcar_unw.item()),
        "mean_weighted_rcar_loss": float(l_rcar_w.item()),
        "loss_ratio_unweighted": float(ratio_unw),
        "loss_ratio_weighted": float(ratio_w),
        "decoder_gradient_norm_e2ad": float(e2ad_dec_grad),
        "decoder_gradient_norm_rcar_unweighted": float(rcar_unw_dec_grad),
        "decoder_gradient_norm_rcar_weighted": float(rcar_w_dec_grad),
        "gradient_norm_ratio_unweighted": float(rcar_unw_dec_grad / e2ad_dec_grad),
        "gradient_norm_ratio_weighted": float(rcar_w_dec_grad / e2ad_dec_grad),
        "lambda_candidates": [0.05, 0.1, 0.5, 1.0],
        "default_lambda": 1.0
    }
    with open("E2AD_RCAR_OCT2017_CORRECTED/audit/loss_scale_analysis.json", "w", encoding="utf-8") as f:
        json.dump(loss_scale_report, f, indent=2)
    print("Saved loss scale analysis to E2AD_RCAR_OCT2017_CORRECTED/audit/loss_scale_analysis.json")
    print("=" * 70)

if __name__ == "__main__":
    run_shared_init_and_loss_scale()
