import os
import numpy as np

# Set writable matplotlib config dir
os.environ['MPLCONFIGDIR'] = os.path.abspath('scratch/mpl')
os.makedirs('scratch/mpl', exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

def generate_comparison():
    # Pick a sample image from OCT2017
    sample_path = "OCT2017/test/NORMAL/NORMAL-10172-1.jpeg"
    if not os.path.exists(sample_path):
        test_dir = "OCT2017/test/NORMAL"
        files = [f for f in os.listdir(test_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
        sample_path = os.path.join(test_dir, files[0])
    
    print(f"Loading sample image: {sample_path}")
    raw_img = Image.open(sample_path).convert('RGB')
    orig_w, orig_h = raw_img.size
    print(f"Original image dimensions: {orig_w} x {orig_h}")
    
    # 1. Author reference preprocessing:
    # Direct Bilinear Resize to (256, 256) followed by CenterCrop(256, 256) [which is a no-op]
    # As implemented in TumCCC/E2AD datasets/dataset.py: A.Resize(256, 256) + A.CenterCrop(256, 256)
    author_transform = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop((256, 256)),
    ])
    author_img = author_transform(raw_img)
    author_np = np.array(author_img)
    
    # 2. Aspect-ratio preserving preprocessing:
    # Resize shorter edge to 256 while preserving aspect ratio, then CenterCrop (256, 256)
    aspect_transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop((256, 256)),
    ])
    aspect_img = aspect_transform(raw_img)
    aspect_np = np.array(aspect_img)
    
    # Difference between squashed (author) and aspect-preserving
    diff_np = np.abs(author_np.astype(float) - aspect_np.astype(float)).mean(axis=-1)
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=300)
    
    axes[0].imshow(raw_img)
    axes[0].set_title(f"1. Original OCT Scan\n({orig_w}x{orig_h} Native Aspect Ratio)", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(author_np)
    axes[1].set_title(f"2. Reference E2AD Preprocessing\n(Direct Resize to 256x256 Squash)", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(aspect_np)
    axes[2].set_title(f"3. Aspect-Preserving Alternative\n(Resize Shorter Side 256 + CenterCrop)", fontsize=11, fontweight='bold')
    axes[2].axis('off')
    
    im_diff = axes[3].imshow(diff_np, cmap='inferno')
    axes[3].set_title(f"4. Spatial Deformation Difference\n(Max Pixel Error: {diff_np.max():.1f})", fontsize=11, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im_diff, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.suptitle("OCT2017 Preprocessing Transformation Audit & Geometric Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    
    out_paths = [
        "figures/preprocessing_comparison.png",
        "E2AD_RCAR_OCT2017_CORRECTED/figures/preprocessing_comparison.png"
    ]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, bbox_inches='tight', dpi=300)
        print(f"Saved figure: {p}")
    plt.close(fig)
    print("Preprocessing comparison figure successfully created!")

if __name__ == "__main__":
    generate_comparison()
