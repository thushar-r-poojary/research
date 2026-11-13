import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from PIL import Image
from models.adat import ADAT
from dataset import ProbavDataset
from enhancer import SatelliteEnhancer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    probav_dir = os.path.join(script_dir, 'data/probav')
    if not os.path.exists(probav_dir):
        probav_dir = 'data/probav'

    # Load scene imgset0000 from real PROBA-V dataset
    ds = ProbavDataset(root_dir=probav_dir, split='train', band='RED', num_frames=9, is_train=False, scale=3)
    batch = ds[0] # imgset0000
    
    lr1 = batch['lr1'].unsqueeze(0).to(device)
    lr2 = batch['lr2'].unsqueeze(0).to(device)
    hr = batch['hr']
    
    print(f"PROBA-V Real Satellite Scene (imgset0000) Loaded!")
    print(f"LR1 Input Shape: {lr1.shape} (128x128 resolution, 300m per pixel)")
    print(f"LR2 Input Shape: {lr2.shape} (128x128 resolution, 300m per pixel)")
    print(f"Ground Truth HR Shape: {hr.shape} (384x384 resolution, 100m per pixel)")

    # Load ADAT model with matching checkpoint params
    embed_dim = 16
    num_blocks = 2
    weights_path = os.path.join(script_dir, 'weights/adat_best.pth')
    config_path = os.path.join(script_dir, 'weights/config.json')
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            cfg = json.load(f)
            embed_dim = cfg.get('embed_dim', embed_dim)
            num_blocks = cfg.get('num_encoder_blocks', num_blocks)

    model = ADAT(in_channels=1, out_channels=1, embed_dim=embed_dim, num_encoder_blocks=num_blocks, scale=3).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Successfully loaded trained ADAT weights from: {weights_path}")

    model.eval()
    with torch.no_grad():
        out = model(lr1, lr2)
        out_clamped = torch.clamp(out, 0, 1)

    print(f"Model Output HR Shape: {out_clamped.shape} (384x384 resolution, 100m per pixel)")

    # Enhanced processing with SatelliteEnhancer
    img_lr1, _ = SatelliteEnhancer.enhance(batch['lr1'], stretch=True, clahe=False, sharpen=False)
    img_lr2, _ = SatelliteEnhancer.enhance(batch['lr2'], stretch=True, clahe=False, sharpen=False)
    img_hr, _  = SatelliteEnhancer.enhance(out_clamped, stretch=True, clahe=True, sharpen=False)
    img_sharp, _ = SatelliteEnhancer.enhance(out_clamped, stretch=True, clahe=True, sharpen=True, sharp_strength=2.2)

    # Save to Dual_Image_SR_Project directory
    img_lr1.save('input_lr1_display.png')
    img_lr2.save('input_lr2_display.png')
    img_hr.save('output_hr.png')
    img_sharp.save('output_hr_sharp.png')

    # Also save to parent Research directory
    img_lr1.save('../input_lr1_display.png')
    img_lr2.save('../input_lr2_display.png')
    img_hr.save('../output_hr.png')
    img_sharp.save('../output_hr_sharp.png')

    print("\nSUCCESS: All enhanced real satellite input and super-resolved HR output display images generated!")

if __name__ == '__main__':
    main()

