import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.adat import ADAT
from models.multi_image import AgenticMultiFrameTransformer
from dataset import ProbavDataset
import json
import argparse

# ─────────────────────────────────────────────────────────────────────────────
# MASKED LOSSES FOR PROBA-V (Computes loss only on cloud-free/valid pixels)
# ─────────────────────────────────────────────────────────────────────────────
class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        loss = F.l1_loss(pred, target, reduction='none')
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)

class MaskedProgressiveLoss(nn.Module):
    """
    Progressive loss transitioning from Masked L1 to Masked MSE (L2) loss.
    L1 loss is used for the first few epochs for stable gradients,
    then L2 is blended in to directly maximize PSNR.
    """
    def __init__(self, mse_warmup_epoch=10):
        super().__init__()
        self.l1      = MaskedL1Loss()
        self.mse     = MaskedMSELoss()
        self.warmup  = mse_warmup_epoch
        self.epoch   = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, pred, target, mask):
        if self.epoch <= self.warmup:
            return self.l1(pred, target, mask)
        progress = min(1.0, (self.epoch - self.warmup) / 10.0)
        return (1.0 - progress) * self.l1(pred, target, mask) + progress * self.mse(pred, target, mask)

def psnr(a, b, mask=None):
    if mask is not None:
        mse = torch.sum(((a - b) * mask) ** 2) / (mask.sum() + 1e-8)
    else:
        mse = torch.mean((a - b) ** 2)
    return 50.0 if mse.item() < 1e-10 else 10 * torch.log10(1.0 / mse).item()

def prepare_data():
    probav_dir = 'data/probav'
    if not os.path.exists(probav_dir):
        print("PROBA-V dataset not found at 'data/probav'. Running download_probav.py first...")
        try:
            import download_probav
            download_probav.main()
        except Exception as e:
            print(f"Could not download dataset: {e}")
    return probav_dir

def train():
    parser = argparse.ArgumentParser(description="Train ADAT or AMST on PROBA-V Dataset")
    parser.add_argument("--model", type=str, default="adat", choices=["adat", "amst"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embed dimension")
    parser.add_argument("--num_blocks", type=int, default=8, help="Number of encoder/fusion blocks")
    parser.add_argument("--patch_size", type=int, default=64, help="LR patch size for training crops")
    args = parser.parse_args()

    scale = 3  # Fixed for PROBA-V

    print("=" * 65)
    print(f"  Training {args.model.upper()} on PROBA-V Satellite Dataset")
    print(f"  Scale:{scale}x  Embed:{args.embed_dim}  Blocks:{args.num_blocks}  Epochs:{args.epochs}  Batch:{args.batch_size}  Patch:{args.patch_size}x{args.patch_size}")
    print("  Loss: Masked L1 -> Masked MSE | Targets: 1-channel RED/NIR reflectances")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probav_dir = prepare_data()
    if not os.path.exists(probav_dir):
        print("Error: Could not obtain dataset. Aborting.")
        return

    # Initialize Dataset with custom crop patch size
    dataset = ProbavDataset(root_dir=probav_dir, split='train', band='RED', num_frames=9,
                            is_train=True, patch_size=args.patch_size, scale=scale)
    
    if len(dataset) == 0:
        print("Error: Dataset is empty. Ensure data/probav contains RED/train or train/RED folder structure.")
        return
        
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset)} scenes -> {len(loader)} batches/epoch")

    # Select and initialize model
    if args.model == "adat":
        model = ADAT(in_channels=1, out_channels=1, embed_dim=args.embed_dim, num_encoder_blocks=args.num_blocks, scale=scale).to(device)
    else:
        model = AgenticMultiFrameTransformer(scale=scale, num_frames=9, num_features=args.embed_dim, num_blocks=args.num_blocks).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} parameters loaded on {device}")

    # Checkpoint configuration
    ckpt_dir = 'weights'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{args.model}_best.pth')

    # Save configuration
    try:
        config = {
            'model': args.model,
            'scale': scale,
            'embed_dim': args.embed_dim,
            'num_encoder_blocks': args.num_blocks if args.model == 'adat' else None,
            'num_blocks': args.num_blocks if args.model == 'amst' else None
        }
        with open(os.path.join(ckpt_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Warning: could not save config.json: {e}")

    # Load checkpoint if exists and matches parameter size
    if os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device)
            # Only load if shapes match (to avoid crash during parameter mismatch)
            model.load_state_dict(state, strict=True)
            print(f"Loaded existing checkpoint from {ckpt_path}")
        except Exception as e:
            print(f"Could not load checkpoint ({e}). Starting training from scratch.")

    criterion = MaskedProgressiveLoss(mse_warmup_epoch=5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_psnr = 0.0
    model.train()

    for epoch in range(1, args.epochs + 1):
        loss_sum, psnr_sum, n = 0.0, 0.0, 0
        criterion.set_epoch(epoch)

        for batch in loader:
            hr  = batch['hr'].to(device)
            qm_hr = batch['qm_hr'].to(device)

            if args.model == "adat":
                lr1 = batch['lr1'].to(device)
                lr2 = batch['lr2'].to(device)
                out = model(lr1, lr2)
            else:
                lr_seq = batch['lr_seq'].to(device)
                out = model(lr_seq)

            # Ensure spatial shapes match
            if out.shape != hr.shape:
                hr = F.interpolate(hr, size=out.shape[2:], mode='bicubic', align_corners=False)
                qm_hr = F.interpolate(qm_hr, size=out.shape[2:], mode='nearest')

            out = torch.clamp(out, 0, 1)
            hr = torch.clamp(hr, 0, 1)

            loss = criterion(out, hr, qm_hr)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += loss.item()
            current_psnr = psnr(out.detach(), hr.detach(), qm_hr)
            psnr_sum += current_psnr
            n += 1

            if n == 1 or n % 10 == 0 or n == len(loader):
                print(f"  Batch [{n:2d}/{len(loader)}]  Loss: {loss.item():.4f}  Masked PSNR: {current_psnr:.2f} dB")

        scheduler.step()
        avg_loss = loss_sum / n
        avg_psnr = psnr_sum / n
        lr_now   = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch:2d}/{args.epochs}]  Loss:{avg_loss:.4f}  Masked PSNR:{avg_psnr:.2f} dB  LR:{lr_now:.2e}")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), ckpt_path)
            print(f"--> Saved best weights! ({best_psnr:.2f} dB)")

    torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{args.model}_latest.pth'))
    print(f"\n{'='*65}")
    print(f"  DONE - Best Masked PSNR: {best_psnr:.2f} dB")
    print(f"  Best Weights saved to: {ckpt_path}")
    print(f"{'='*65}")

if __name__ == '__main__':
    train()
