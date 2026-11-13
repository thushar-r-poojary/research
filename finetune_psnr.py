import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
sys.stdout.reconfigure(line_buffering=True)

import torch
torch.set_num_threads(2)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.adat import ADAT
from train import DualSRDataset, prepare_data, psnr
import json

# ─────────────────────────────────────────────────────────────────────────────
# FINE-TUNING CONFIG FOR MAX PSNR
# ─────────────────────────────────────────────────────────────────────────────
SCALE      = 4        # 4x SR
EMBED_DIM  = 96       
NUM_BLOCKS = 12       
EPOCHS     = 5        # 5 epochs is perfect for quick CPU fine-tuning
LR_INIT    = 5e-5     # Low learning rate for fine-tuning
BATCH_SIZE = 2        # Small batch size to avoid CPU RAM allocation limits
PATCH_HR   = 128      
PATCH_LR   = PATCH_HR // SCALE

def finetune():
    print("=" * 65)
    print(f"  ADAT {SCALE}× SR — PSNR Boost (Fine-tuning with MSE loss)")
    print(f"  Scale:{SCALE}×  Embed:{EMBED_DIM}  Blocks:{NUM_BLOCKS}  Epochs:{EPOCHS}")
    print(f"  Loss: Pure MSE (L2) to directly maximize PSNR | LR:{LR_INIT}")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr1_dir, lr2_dir, hr_dir = prepare_data()
    if lr1_dir is None:
        return

    dataset = DualSRDataset(lr1_dir, lr2_dir, hr_dir,
                            patch_lr=PATCH_LR, scale=SCALE, is_train=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset)} images -> {len(loader)} batches/epoch")

    # Model
    model = ADAT(embed_dim=EMBED_DIM, num_encoder_blocks=NUM_BLOCKS, scale=SCALE).to(device)
    
    # Load the best checkpoint to start fine-tuning
    ckpt = 'weights/adat_best.pth'
    if os.path.exists(ckpt):
        try:
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state)
            print(f"Successfully loaded best checkpoint: {ckpt} for fine-tuning.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Cannot proceed with fine-tuning.")
            return
    else:
        print(f"No checkpoint found at {ckpt}. Run train.py first.")
        return

    # Direct PSNR optimization using MSE loss (L2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR_INIT, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Initialize best crop PSNR to the best value from training
    best_psnr = 0.0
    model.train()

    for epoch in range(1, EPOCHS + 1):
        loss_sum, psnr_sum, n = 0.0, 0.0, 0

        for batch in loader:
            lr1 = batch['lr1'].to(device)
            lr2 = batch['lr2'].to(device)
            hr  = batch['hr'].to(device)

            out = model(lr1, lr2)
            if out.shape != hr.shape:
                hr = F.interpolate(hr, size=out.shape[2:], mode='bicubic', align_corners=False)
            out = torch.clamp(out, 0, 1)

            loss = criterion(out, hr)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += loss.item()
            current_psnr = psnr(out.detach(), hr.detach())
            psnr_sum += current_psnr
            n += 1

            if n == 1 or n % 5 == 0 or n == len(loader):
                print(f"  Batch [{n:2d}/{len(loader)}]  MSE Loss: {loss.item():.6f}  PSNR: {current_psnr:.2f} dB")

        scheduler.step()
        avg_loss = loss_sum / n
        avg_psnr = psnr_sum / n
        lr_now   = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch:3d}/{EPOCHS}]  Avg MSE Loss:{avg_loss:.6f}  Avg PSNR:{avg_psnr:.2f} dB  LR:{lr_now:.2e}")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), 'weights/adat_best.pth')
            print(f"--> Saved new best fine-tuned weights! (Avg PSNR: {avg_psnr:.2f} dB)")

    torch.save(model.state_dict(), 'weights/adat_latest.pth')
    print(f"\n{'='*65}")
    print(f"  FINE-TUNING DONE - Best Crop PSNR: {best_psnr:.2f} dB")
    print(f"  Weights updated: weights/adat_best.pth")
    print(f"{'='*65}")

if __name__ == '__main__':
    finetune()
