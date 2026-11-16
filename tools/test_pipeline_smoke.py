import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from tools.test_architecture_equivalence import E2AD_Unified
from tools.test_rcar_validation import RCARLoss
from train_utils import get_optimizer_v2, get_multistep_schedule_with_warmup

class SimpleOCTDataset(Dataset):
    def __init__(self, file_paths, targets):
        self.file_paths = file_paths
        self.targets = targets
        self.transform = T.Compose([
            T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img = Image.open(path).convert('RGB')
        tensor = self.transform(img)
        return tensor, self.targets[idx], os.path.basename(path)

def run_smoke_test():
    print("=" * 70)
    print("PHASE 8: CONTROLLED PIPELINE SMOKE TEST")
    print("=" * 70)
    
    device = torch.device('cpu')
    train_dir = "OCT2017/train/NORMAL"
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)[:32]]
    train_dset = SimpleOCTDataset(train_files, [0]*len(train_files))
    
    # 40 validation samples (10 per class)
    val_files, val_targets = [], []
    for cls, target in [("NORMAL", 0), ("CNV", 1), ("DME", 1), ("DRUSEN", 1)]:
        c_dir = os.path.join("OCT2017/test", cls)
        c_files = [os.path.join(c_dir, f) for f in os.listdir(c_dir)[:10]]
        val_files.extend(c_files)
        val_targets.extend([target]*len(c_files))
    val_dset = SimpleOCTDataset(val_files, val_targets)
    val_loader = DataLoader(val_dset, batch_size=8, shuffle=False)
    
    # Arm configurations
    arms = [
        {"name": "Arm_A_Baseline", "mode": "none", "alpha": 0.0, "lambda": 0.0},
        {"name": "Arm_B_Unweighted_RCAR", "mode": "unweighted", "alpha": 0.0, "lambda": 1.0},
        {"name": "Arm_C_ConfWeighted_RCAR", "mode": "weighted", "alpha": 2.0, "lambda": 1.0}
    ]
    
    init_state_path = "initial_model_state.pth"
    assert os.path.exists(init_state_path), "Missing initial_model_state.pth"
    
    for arm in arms:
        print(f"\n--- Testing {arm['name']} (mode={arm['mode']}, lambda={arm['lambda']}) ---")
        
        # Load EXACT shared initial state
        model = E2AD_Unified(bn_pretrain=True).to(device)
        model.load_state_dict(torch.load(init_state_path, map_location=device))
        
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.01
                
        opt = get_optimizer_v2(model, 'AdamW', lr=5e-4, momentum=0.9, lr_encoder=1e-5, weight_decay=1e-4)
        sch = get_multistep_schedule_with_warmup(opt, milestones=[100000], gamma=0.2, num_warmup_steps=0)
        rcar_loss_fn = RCARLoss(mode=arm['mode'], alpha=arm['alpha'])
        
        # Deterministic batch ordering
        g = torch.Generator().manual_seed(42)
        train_loader = DataLoader(train_dset, batch_size=4, shuffle=True, generator=g)
        
        model.train()
        train_iter = iter(train_loader)
        
        for it in range(3):
            try:
                x_b, _, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_b, _, _ = next(train_iter)
                
            out = model(x_b)
            l_e2ad = out['loss']
            
            if arm['mode'] == 'none':
                l_rcar = torch.tensor(0.0)
                total_loss = l_e2ad
            else:
                l_rcar, rcar_diag, _ = rcar_loss_fn(out['A1'], out['A2'])
                total_loss = l_e2ad + arm['lambda'] * l_rcar
                
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            sch.step()
            
            print(f"  Iter {it+1}/3 | Total Loss: {total_loss.item():.4f} (E2AD: {l_e2ad.item():.4f}, RCAR: {l_rcar.item():.4f})")
            
        # Evaluation
        model.eval()
        all_scores_max = []
        all_scores_mean = []
        all_targets = []
        
        with torch.no_grad():
            for x_v, y_v, _ in val_loader:
                res = model(x_v)
                p1 = res['p_all_1']
                p2 = res['p_all_2']
                # Max reduction (paper)
                score_max = ((p1.flatten(1).max(1)[0] + p2.flatten(1).max(1)[0]) / 2.0).tolist()
                # Mean reduction
                score_mean = ((p1.flatten(1).mean(1) + p2.flatten(1).mean(1)) / 2.0).tolist()
                
                all_scores_max.extend(score_max)
                all_scores_mean.extend(score_mean)
                all_targets.extend(y_v.tolist())
                
        auc_max = roc_auc_score(all_targets, all_scores_max)
        auc_mean = roc_auc_score(all_targets, all_scores_mean)
        print(f"  Smoke Val AUROC (Max mode ): {auc_max*100:.2f}%")
        print(f"  Smoke Val AUROC (Mean mode): {auc_mean*100:.2f}%")
        
    print("\n" + "=" * 70)
    print("PHASE 8 SMOKE TEST PASSED COMPLETELY WITHOUT ERRORS!")
    print("=" * 70)

if __name__ == "__main__":
    run_smoke_test()
