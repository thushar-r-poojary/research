import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import json
import time
from models.adat import ADAT

def export_adat_torchscript(output_path="weights/adat_traced.pt"):
    """
    Exports ADAT to standalone TorchScript format for C++, embedded,
    and satellite edge computing deployments without Python dependencies.
    """
    device = torch.device('cpu')
    
    # Load configuration
    config_path = "weights/config.json"
    scale = 3
    embed_dim = 16
    num_encoder_blocks = 2
    in_channels = 1
    out_channels = 1
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
                scale = cfg.get("scale", scale)
                embed_dim = cfg.get("embed_dim", embed_dim)
                num_encoder_blocks = cfg.get("num_encoder_blocks", num_encoder_blocks)
        except Exception as e:
            print(f"Warning: could not load config: {e}")

    print("=" * 65)
    print("  Exporting ADAT to Standalone TorchScript for Edge Deployment")
    print(f"  Scale: {scale}x | Embed Dim: {embed_dim} | Blocks: {num_encoder_blocks}")
    print("=" * 65)

    # Initialize model
    model = ADAT(in_channels=in_channels, out_channels=out_channels, 
                 embed_dim=embed_dim, num_encoder_blocks=num_encoder_blocks, scale=scale).to(device)

    weights_path = "weights/adat_best.pth"
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Warning: could not load weights strictly: {e}")

    model.eval()

    # Create dummy tensors for tracing: (B, C, H, W)
    dummy_lr1 = torch.randn(1, in_channels, 64, 64, device=device)
    dummy_lr2 = torch.randn(1, in_channels, 64, 64, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Tracing model execution and compiling...")
    traced_model = torch.jit.trace(model, (dummy_lr1, dummy_lr2))
    torch.jit.save(traced_model, output_path)

    size_kb = os.path.getsize(output_path) / 1024.0
    print(f"Export successful! File: {output_path} ({size_kb:.1f} KB)")

    # Verify reload of traced model
    reloaded = torch.jit.load(output_path, map_location=device)
    reloaded.eval()
    with torch.no_grad():
        out = reloaded(dummy_lr1, dummy_lr2)
    print(f"Verification passed! Output tensor shape: {out.shape}")

    # Benchmark CPU latency
    print("\nBenchmarking CPU inference latency (20 iterations)...")
    with torch.no_grad():
        for _ in range(5):  # Warmup
            _ = reloaded(dummy_lr1, dummy_lr2)
        start = time.perf_counter()
        for _ in range(20):
            _ = reloaded(dummy_lr1, dummy_lr2)
        elapsed = (time.perf_counter() - start) / 20.0 * 1000.0

    print(f"Average CPU Latency: {elapsed:.2f} ms per 64x64 patch")
    print("=" * 65)

if __name__ == "__main__":
    export_adat_torchscript()

