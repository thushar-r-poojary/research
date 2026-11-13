import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
from dataset import ProbavDataset
from models.adat import ADAT
from models.multi_image import AgenticMultiFrameTransformer
import numpy as np
import json
import argparse

# Default configuration (will be updated dynamically if config.json is found)
SCALE = 3
EMBED_DIM = 64
NUM_BLOCKS = 8
MODEL_TYPE = 'adat'

config_path = "weights/config.json"
if os.path.exists(config_path):
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
            SCALE = cfg.get("scale", SCALE)
            EMBED_DIM = cfg.get("embed_dim", EMBED_DIM)
            # Support both config formats
            NUM_BLOCKS = cfg.get("num_encoder_blocks", cfg.get("num_blocks", NUM_BLOCKS))
            MODEL_TYPE = cfg.get("model", MODEL_TYPE)
            print(f"Loaded config from {config_path}: model={MODEL_TYPE}, scale={SCALE}, embed_dim={EMBED_DIM}, blocks={NUM_BLOCKS}")
    except Exception as e:
        print(f"Warning: could not load config from {config_path}: {e}")

def compute_psnr(pred, target, mask=None):
    if mask is not None:
        mse = torch.sum(((pred - target) * mask) ** 2) / (mask.sum() + 1e-8)
    else:
        mse = torch.mean((pred - target) ** 2)
    return 50.0 if mse.item() < 1e-10 else 10 * torch.log10(1.0 / mse).item()

def compute_ssim(pred, gt, mask=None, C1=0.01**2, C2=0.03**2):
    """Compute a channel-wise mean SSIM over valid pixels."""
    if mask is not None:
        mu_p = (pred * mask).sum() / (mask.sum() + 1e-8)
        mu_g = (gt * mask).sum() / (mask.sum() + 1e-8)
        var_p = (((pred - mu_p) * mask) ** 2).sum() / (mask.sum() + 1e-8)
        var_g = (((gt - mu_g) * mask) ** 2).sum() / (mask.sum() + 1e-8)
        cov = ((pred - mu_p) * (gt - mu_g) * mask).sum() / (mask.sum() + 1e-8)
    else:
        mu_p = pred.mean()
        mu_g = gt.mean()
        var_p = pred.var()
        var_g = gt.var()
        cov = ((pred - mu_p) * (gt - mu_g)).mean()
        
    ssim = ((2*mu_p*mu_g + C1) * (2*cov + C2)) / ((mu_p**2 + mu_g**2 + C1) * (var_p + var_g + C2))
    return ssim.item()

def evaluate_all():
    parser = argparse.ArgumentParser(description="Evaluate ADAT or AMST on PROBA-V")
    parser.add_argument("--model", type=str, default=MODEL_TYPE, choices=["adat", "amst"], help="Model type to evaluate")
    parser.add_argument("--num_images", type=int, default=30, help="Number of scenes to evaluate")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize appropriate model
    if args.model == "adat":
        model = ADAT(in_channels=1, out_channels=1, embed_dim=EMBED_DIM, num_encoder_blocks=NUM_BLOCKS, scale=SCALE).to(device)
        weights_path = "weights/adat_best.pth"
    else:
        model = AgenticMultiFrameTransformer(scale=SCALE, num_frames=9, num_features=EMBED_DIM, num_blocks=NUM_BLOCKS).to(device)
        weights_path = "weights/amst_best.pth"

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: No weights found at {weights_path}. Running with randomly initialized model.")

    model.eval()

    # Load dataset (evaluating on the first few scenes of the 'train' split which have HR ground truth)
    ds = ProbavDataset(root_dir='data/probav', split='train', band='RED', num_frames=9, is_train=False, scale=SCALE)
    
    if len(ds) == 0:
        print("Error: No scenes found in data/probav/train/RED. Run download_probav.py first.")
        return

    num_eval = min(args.num_images, len(ds))
    total_psnr = 0.0
    total_ssim = 0.0
    valid_images = 0

    print("=" * 70)
    print(f"  PROBA-V Evaluation — Model: {args.model.upper()} ({SCALE}x SR) — {num_eval} Scenes")
    print("  Note: PSNR and SSIM are calculated strictly on cloud-free pixels (unboosted)")
    print("=" * 70)

    results = []
    for idx in range(num_eval):
        batch = ds[idx]
        filename = batch['filename']
        hr = batch['hr'].unsqueeze(0).to(device)       # (1, 1, HR_H, HR_W)
        qm_hr = batch['qm_hr'].unsqueeze(0).to(device) # (1, 1, HR_H, HR_W)

        if args.model == "adat":
            lr1 = batch['lr1'].unsqueeze(0).to(device) # (1, 1, H, W)
            lr2 = batch['lr2'].unsqueeze(0).to(device) # (1, 1, H, W)
            with torch.no_grad():
                hr_output = model(lr1, lr2)
        else:
            lr_seq = batch['lr_seq'].unsqueeze(0).to(device) # (1, 9, H, W)
            with torch.no_grad():
                hr_output = model(lr_seq)

        hr_output_clamped = torch.clamp(hr_output, 0, 1)

        # Interpolate ground truth target to match output size if needed
        if gt_shape := hr_output_clamped.shape[2:]:
            if hr.shape[2:] != gt_shape:
                hr = F.interpolate(hr, size=gt_shape, mode='bicubic', align_corners=False)
                qm_hr = F.interpolate(qm_hr, size=gt_shape, mode='nearest')

        hr = torch.clamp(hr, 0, 1)

        psnr_val = compute_psnr(hr_output_clamped, hr, qm_hr)
        ssim_val = compute_ssim(hr_output_clamped, hr, qm_hr)

        total_psnr += psnr_val
        total_ssim += ssim_val
        valid_images += 1

        line = f"[{idx+1:3d}/{num_eval}] {filename} -> True PSNR: {psnr_val:.2f} dB  True SSIM: {ssim_val:.4f}"
        print(line)
        results.append(line)

    if valid_images == 0:
        print("No valid images evaluated.")
        return

    avg_psnr = total_psnr / valid_images
    avg_ssim = total_ssim / valid_images
    summary = (
        f"\nPROBA-V Dataset Summary ({args.model.upper()}):\n"
        f"Evaluated Scenes: {valid_images}\n"
        f"Average True PSNR: {avg_psnr:.2f} dB\n"
        f"Average True SSIM: {avg_ssim:.4f}"
    )
    print("-" * 70)
    print(summary)
    print("=" * 70)

    with open("psnr_results.txt", "w") as f:
        f.write("\n".join(results) + "\n\n" + summary + "\n")
    print("Results saved to: psnr_results.txt")

if __name__ == "__main__":
    evaluate_all()
