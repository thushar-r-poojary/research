import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Standard residual block without BatchNorm (EDSR-style)."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.scale = 0.1  # Residual scaling for stability

    def forward(self, x):
        return x + self.block(x) * self.scale


class TransformerEncoder(nn.Module):
    """
    Deep Feature Extractor with 16 Residual Blocks (EDSR-style).
    Extracts rich multi-scale features from a single LR image.
    """
    def __init__(self, in_channels=3, embed_dim=64, num_blocks=16):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # Deep residual body
        body = [ResidualBlock(embed_dim) for _ in range(num_blocks)]
        body.append(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        feat = self.conv_first(x)
        deep = self.body(feat)
        return feat + deep  # Global residual connection
