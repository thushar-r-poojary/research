import os
# Bypass OpenMP conflict in local execution
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from models.adat import ADAT
from evaluation.agentic_eval import AgenticEvaluator
import json

def process_image(img_path):
    """Load an image and convert to tensor."""
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

def main():
    print("=" * 60)
    print("  ADAT Dual-Image Interactive Terminal Inference")
    print("=" * 60)
    
    # Prompt user for dual input file paths
    lr1_path = input("Enter path to the first input low-resolution image (lr1): ").strip()
    if not os.path.exists(lr1_path):
        print(f"Error: File '{lr1_path}' does not exist.")
        return

    lr2_path = input("Enter path to the second input low-resolution image (lr2): ").strip()
    if not os.path.exists(lr2_path):
        print(f"Error: File '{lr2_path}' does not exist.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    scale = 4
    embed_dim = 64
    num_encoder_blocks = 8

    config_path = "weights/config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
                scale = cfg.get("scale", scale)
                embed_dim = cfg.get("embed_dim", embed_dim)
                num_encoder_blocks = cfg.get("num_encoder_blocks", num_encoder_blocks)
        except Exception as e:
            print(f"Warning: could not load config from {config_path}: {e}")

    # Load model
    print(f"\nLoading ADAT model ({scale}x SR)...")
    model = ADAT(embed_dim=embed_dim, num_encoder_blocks=num_encoder_blocks, scale=scale).to(device)
    weights_path = "weights/adat_best.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        print(f"Loaded weights: {weights_path}")
    else:
        print("Warning: Weights file not found! Running with random weights.")
    model.eval()
    
    # Run processing
    print(f"\nProcessing images: {lr1_path} and {lr2_path}...")
    lr1_tensor = process_image(lr1_path).to(device)
    lr2_tensor = process_image(lr2_path).to(device)
    
    with torch.no_grad():
        hr_output = model(lr1_tensor, lr2_tensor)
        hr_output_clamped = torch.clamp(hr_output, 0, 1)
        
    out_path = "output_interactive_hr.png"
    save_image(hr_output_clamped, out_path)
    print(f"\nHR image successfully generated and saved to: {out_path}")
    
    # Run evaluation
    print("\nRunning Agentic Blind Evaluator...")
    evaluator = AgenticEvaluator()
    report = evaluator.evaluate(hr_output_clamped)
    print(report)

if __name__ == '__main__':
    main()
