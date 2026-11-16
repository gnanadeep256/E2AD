import os
import sys
sys.path.insert(0, os.path.abspath('.'))
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

class RCARLoss(nn.Module):
    def __init__(self, mode: str = "weighted", alpha: float = 2.0, epsilon: float = 1e-7):
        super(RCARLoss, self).__init__()
        assert mode in ["none", "unweighted", "weighted"], f"Unsupported mode: {mode}"
        self.mode = mode
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)

    def forward(self, A1: torch.Tensor, A2: torch.Tensor):
        if self.mode == "none":
            zero_loss = torch.tensor(0.0, device=A1.device, dtype=A1.dtype)
            return zero_loss, {
                "loss_rcar": 0.0,
                "unweighted_cons": 0.0,
                "weighted_cons": 0.0,
                "w_mean": 1.0,
                "w_min": 1.0,
                "w_max": 1.0,
                "res_mean_1": float(A1.detach().mean().item()),
                "res_mean_2": float(A2.detach().mean().item())
            }

        diff = torch.abs(A1 - A2)
        unweighted_cons = diff.mean()

        if self.mode == "unweighted":
            loss = unweighted_cons
            w_mean, w_min, w_max = 1.0, 1.0, 1.0
            weighted_cons = float(unweighted_cons.item())
            w_tensor = torch.ones_like(diff)
        elif self.mode == "weighted":
            sum_res = (A1 + A2).detach() # stop_gradient
            w = torch.exp(-self.alpha * sum_res)
            denom = torch.sum(w) + self.epsilon
            loss = torch.sum(w * diff) / denom
            w_mean = float(w.mean().item())
            w_min = float(w.min().item())
            w_max = float(w.max().item())
            weighted_cons = float(loss.item())
            w_tensor = w

        metrics = {
            "loss_rcar": float(loss.item()),
            "unweighted_cons": float(unweighted_cons.item()),
            "weighted_cons": weighted_cons,
            "w_mean": w_mean,
            "w_min": w_min,
            "w_max": w_max,
            "res_mean_1": float(A1.detach().mean().item()),
            "res_mean_2": float(A2.detach().mean().item())
        }
        return loss, metrics, w_tensor

def validate_rcar():
    print("=" * 70)
    print("R-CAR FORMULATION & MATHEMATICAL VALIDATION")
    print("=" * 70)
    
    device = torch.device('cpu')
    torch.manual_seed(42)
    
    # 1. Coordinate alignment verification with asymmetric landmark grid
    H, W = 64, 64
    grid = torch.zeros(1, 1, H, W)
    # Place asymmetric landmark at top-left corner (r=10, c=15)
    grid[0, 0, 10, 15] = 100.0
    
    # Rotate 180 degrees
    grid_rot = torch.rot90(grid, 2, dims=(-2, -1))
    # In rotated frame, landmark should be at (H-1-10, W-1-15) = (53, 48)
    assert grid_rot[0, 0, 53, 48] == 100.0, "180 deg rotation landmark mismatch"
    
    # Inverse rotate back to original frame
    grid_restored = torch.rot90(grid_rot, 2, dims=(-2, -1))
    grid_diff = torch.abs(grid - grid_restored).max().item()
    assert grid_diff == 0.0, f"Inverse coordinate alignment error: {grid_diff}"
    print("1. Coordinate Landmark Alignment Check : PASS (Exact pixel-level inverse match)")
    
    # 2. Instantiate E2AD_Unified model
    model = E2AD_Unified(bn_pretrain=True).to(device)
    model.train()
    
    x = torch.randn(2, 3, 256, 256, device=device)
    out = model(x)
    A1 = out['A1']
    A2 = out['A2']
    
    # Check 1: Identical spatial dimensions
    assert A1.shape == A2.shape == (2, 1, 64, 64), f"Dimension mismatch: A1={A1.shape}, A2={A2.shape}"
    print("2. Spatial Dimension Equality Check   : PASS (Both [2, 1, 64, 64])")
    
    # Check 2: No NaN or Inf
    assert not torch.isnan(A1).any() and not torch.isinf(A1).any(), "NaN/Inf in A1"
    assert not torch.isnan(A2).any() and not torch.isinf(A2).any(), "NaN/Inf in A2"
    print("3. Finite Residual Map Check         : PASS (Zero NaN or Inf detected)")
    
    # Check 3: Alpha sweep (0, 1, 2, 4)
    alpha_results = {}
    weights_dict = {}
    alphas = [0.0, 1.0, 2.0, 4.0]
    
    for a in alphas:
        loss_fn = RCARLoss(mode="weighted", alpha=a)
        l, m, w = loss_fn(A1, A2)
        alpha_results[f"alpha_{a}"] = m
        weights_dict[a] = w.detach().cpu().numpy().flatten()
        print(f"   Alpha = {a:3.1f} | Loss: {m['loss_rcar']:.6f} | w_mean: {m['w_mean']:.4f} | w_min: {m['w_min']:.4f} | w_max: {m['w_max']:.4f}")
    
    # Check 4: For alpha=0, weighted formulation reduces to unweighted formulation
    unweighted_fn = RCARLoss(mode="unweighted")
    l_unw, m_unw, _ = unweighted_fn(A1, A2)
    alpha0_diff = abs(alpha_results["alpha_0.0"]["loss_rcar"] - m_unw["loss_rcar"])
    assert alpha0_diff < 1e-6, f"Alpha=0 did not match unweighted loss! Diff: {alpha0_diff}"
    print(f"4. Alpha=0 Reduction to Unweighted   : PASS (Diff = {alpha0_diff:.2e})")
    
    # Check 5: Denominator safety check with zeros
    zero_A1 = torch.zeros_like(A1)
    zero_A2 = torch.zeros_like(A2)
    l_zero, m_zero, _ = RCARLoss(mode="weighted", alpha=2.0)(zero_A1, zero_A2)
    assert not torch.isnan(l_zero) and not torch.isinf(l_zero), "Denominator instability on zero residuals"
    print("5. Zero-Residual Denominator Safety   : PASS (Loss is finite: 0.0)")
    
    # Check 6: Gradient backprop reachability report
    model.zero_grad()
    l_rcar, _, _ = RCARLoss(mode="weighted", alpha=2.0)(A1, A2)
    l_total = out['loss'] + 1.0 * l_rcar
    l_total.backward()
    
    grad_norms = {}
    modules_to_check = {
        "encoder": model.edc_encoder,
        "decoder_1": model.edc_decoder_1,
        "decoder_2": model.edc_decoder_2,
        "sa1": model.sa1,
        "sa2": model.sa2,
        "sa3": model.sa3,
        "sa4": model.sa4
    }
    
    for name, mod in modules_to_check.items():
        norms = [p.grad.norm().item() for p in mod.parameters() if p.grad is not None]
        total_norm = float(np.sqrt(sum(n**2 for n in norms))) if norms else 0.0
        grad_norms[name] = total_norm
        print(f"   Gradient Norm [{name:<9}]: {total_norm:.4e}")
        assert total_norm > 0, f"Zero gradient on {name}!"
    print("6. Full Gradient Flow & Reachability : PASS (All modules receive active gradients)")
    
    # Check 7: Detached residual maps test
    # Confirm that A1 and A2 are NOT accidentally detached
    assert A1.requires_grad, "A1 has requires_grad=False (Accidentally detached!)"
    assert A2.requires_grad, "A2 has requires_grad=False (Accidentally detached!)"
    print("7. Non-Detached Residual Maps Check   : PASS (A1 and A2 retain full autograd graph)")
    
    # Plot weight distributions across alphas
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
    for i, a in enumerate(alphas):
        w_vals = weights_dict[a]
        axes[i].hist(w_vals, bins=50, color='#2ca02c', alpha=0.7, edgecolor='black')
        axes[i].set_title(f"Weight Distribution (alpha = {a})\nMean: {w_vals.mean():.3f}, Min: {w_vals.min():.3f}", fontsize=10)
        axes[i].set_xlabel("Confidence Weight w(u, v)")
        axes[i].set_ylabel("Pixel Frequency")
        axes[i].grid(True, linestyle='--', alpha=0.5)
    plt.suptitle("R-CAR Confidence Weight Spatial Distribution Under Various Alpha Hyperparameters", fontsize=12, y=1.02)
    plt.tight_layout()
    
    fig_path1 = "figures/rcar_weight_distribution.png"
    fig_path2 = "E2AD_RCAR_OCT2017_CORRECTED/figures/rcar_weight_distribution.png"
    for p in [fig_path1, fig_path2]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved weight distribution plot: {fig_path1}")
    
    # Save reports
    unit_tests_report = {
        "coordinate_inverse_alignment_pass": True,
        "spatial_dimensions": list(A1.shape),
        "zero_nan_or_inf": True,
        "alpha_sweep": alpha_results,
        "alpha_0_reduction_match": float(alpha0_diff),
        "zero_residual_stability": True,
        "a1_requires_grad": True,
        "a2_requires_grad": True,
        "status": "ALL_UNIT_TESTS_PASSED"
    }
    with open("rcar_unit_tests.json", "w", encoding="utf-8") as f:
        json.dump(unit_tests_report, f, indent=2)
    with open("E2AD_RCAR_OCT2017_CORRECTED/audit/rcar_unit_tests.json", "w", encoding="utf-8") as f:
        json.dump(unit_tests_report, f, indent=2)
        
    grad_report = {
        "gradient_norms_by_module": grad_norms,
        "all_target_modules_active": True,
        "status": "GRADIENT_VERIFICATION_PASSED"
    }
    with open("rcar_gradient_report.json", "w", encoding="utf-8") as f:
        json.dump(grad_report, f, indent=2)
    with open("E2AD_RCAR_OCT2017_CORRECTED/audit/rcar_gradient_report.json", "w", encoding="utf-8") as f:
        json.dump(grad_report, f, indent=2)
        
    print("=" * 70)
    print("R-CAR UNIT TESTS AND GRADIENT REPORTS SUCCESSFULLY GENERATED!")
    print("=" * 70)

if __name__ == "__main__":
    validate_rcar()
