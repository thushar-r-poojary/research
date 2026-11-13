import torch
import torch.nn as nn
import torch.nn.functional as F

class AgenticMultiFrameTransformer(nn.Module):
    """
    Agentic Multi-Frame Spectral Transformer (AMST)
    Adapted for the PROBA-V dataset (scale 3, single-channel RED/NIR bands).
    Takes a sequence of low-resolution frames (e.g. 9 frames) and predicts
    the high-resolution ground truth.
    """
    def __init__(self, scale=3, num_frames=9, num_features=64, num_blocks=4):
        super().__init__()
        self.scale = scale
        self.num_frames = num_frames
        
        # input has `num_frames` channels (each frame is a single-channel grayscale image)
        self.feature_extractor = nn.Conv2d(num_frames, num_features, kernel_size=3, padding=1)
        
        # Deep Residual Fusion Blocks to denoise and fuse frames
        self.res_blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
            ) for _ in range(num_blocks)
        ])
        
        # Upsampling block mapping to 1 channel * (scale^2)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, 1 * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        
        # Zero-initialize the upsample convolution weights/bias for stable residual learning
        nn.init.zeros_(self.upsample[0].weight)
        if self.upsample[0].bias is not None:
            nn.init.zeros_(self.upsample[0].bias)

    def forward(self, lr_seq):
        # lr_seq is shape (B, num_frames, H, W)
        
        # Base reconstruction: bicubic interpolation of the first frame (reference frame)
        # In PROBA-V, the first frame is typically the clearest or the primary reference.
        base_hr = F.interpolate(lr_seq[:, :1, :, :], scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        # 1. Feature Extraction
        x = self.feature_extractor(lr_seq)
        
        # 2. Deep Denoising & Fusion
        residual = x
        for block in self.res_blocks:
            x = x + block(x)
        x = x + residual
        
        # 3. Upsampling to HR Space
        out = self.upsample(x)
        
        return base_hr + out
