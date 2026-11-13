import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
from dataset import ProbavDataset
from models.adat import ADAT
from models.multi_image import AgenticMultiFrameTransformer
from torch.utils.data import DataLoader
import json

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
            NUM_BLOCKS = cfg.get("num_encoder_blocks", cfg.get("num_blocks", NUM_BLOCKS))
            MODEL_TYPE = cfg.get("model", MODEL_TYPE)
    except Exception as e:
        print(f"Warning: could not load config from {config_path}: {e}")

def compute_psnr(pred, target, mask=None):
    if mask is not None:
        mse = torch.sum(((pred - target) * mask) ** 2) / (mask.sum() + 1e-8)
    else:
        mse = torch.mean((pred - target) ** 2)
    return 50.0 if mse.item() < 1e-10 else 10 * torch.log10(1.0 / mse).item()

def evaluate():
    device = torch.device('cpu')
    print(f"Loading {MODEL_TYPE.upper()} model...")
    
    if MODEL_TYPE == "adat":
        model = ADAT(in_channels=1, out_channels=1, embed_dim=EMBED_DIM, num_encoder_blocks=NUM_BLOCKS, scale=SCALE).to(device)
        weights_path = 'weights/adat_best.pth'
    else:
        model = AgenticMultiFrameTransformer(scale=SCALE, num_frames=9, num_features=EMBED_DIM, num_blocks=NUM_BLOCKS).to(device)
        weights_path = 'weights/amst_best.pth'

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights: {weights_path}")
    else:
        print(f"Warning: No best weights found at {weights_path}, running initialized weights.")
        
    model.eval()

    print("Loading PROBA-V dataset...")
    ds = ProbavDataset(root_dir='data/probav', split='train', band='RED', num_frames=9, is_train=False, scale=SCALE)
    
    if len(ds) == 0:
        print("Error: No scenes found in data/probav/train/RED. Run download_probav.py first.")
        return
        
    dl = DataLoader(ds, batch_size=4, shuffle=False)

    print("Evaluating first 16 scenes (4 batches x 4)...")
    psnr_sum, n = 0, 0
    with torch.no_grad():
        for batch in dl:
            hr = batch['hr'].to(device)
            qm_hr = batch['qm_hr'].to(device)
            
            if MODEL_TYPE == "adat":
                out = model(batch['lr1'].to(device), batch['lr2'].to(device))
            else:
                out = model(batch['lr_seq'].to(device))

            if out.shape != hr.shape:
                hr = F.interpolate(hr, size=out.shape[2:], mode='bicubic', align_corners=False)
                qm_hr = F.interpolate(qm_hr, size=out.shape[2:], mode='nearest')
                
            out = torch.clamp(out, 0, 1)
            hr = torch.clamp(hr, 0, 1)

            p = compute_psnr(out, hr, qm_hr)
            psnr_sum += p
            n += 1
            print(f"  Batch [{n}/4] Masked PSNR: {p:.2f} dB")
            if n >= 4:
                break

    avg_psnr = psnr_sum / n
    print(f"Validation Average PSNR ({SCALE}x SR on PROBA-V): {avg_psnr:.2f} dB")

if __name__ == '__main__':
    evaluate()
