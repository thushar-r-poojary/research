import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fusion import SpatialPooledCrossAttentionFusion


class ResidualChannelAttentionBlock(nn.Module):
    """
    Residual Channel Attention Block (RCAB) — RCAN style.
    Standard residual block + squeeze-excite channel attention.
    No BatchNorm (EDSR-style for stability).
    """
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(1, ch // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, ch // reduction), ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.body(x)
        res = res * self.ca(res)
        return x + res * 0.1  # residual scaling for training stability


class ADAT(nn.Module):
    """
    ADAT — Agentic Dual-Attention Transformer
    Novel Dual-Image SR architecture with Channel-wise Cross-Attention Fusion (CAF).

    Dual-Image Pipeline:
        LR1 --[Shared RCAB Encoder]--> F1 --|
                                             |--> [CAF Cross-Attention] --> Fused --> [Upsampler] --> HR
        LR2 --[Shared RCAB Encoder]--> F2 --|

    Supports 2x and 4x scale factors.
    Default: embed_dim=64, num_encoder_blocks=8, scale=4 (research-target)
    """
    def __init__(self, in_channels=1, out_channels=1, embed_dim=64, num_encoder_blocks=8, scale=3):
        super().__init__()
        self.scale = scale

        # ── Shared deep encoder (shared weights = implicit regularisation) ──
        self.head = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
        encoder_blocks = [ResidualChannelAttentionBlock(embed_dim) for _ in range(num_encoder_blocks)]
        self.body = nn.Sequential(*encoder_blocks)
        self.body_tail = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # ── Novel: Spatial-Pooled Cross-Attention Fusion (SPCAF) ──
        # Pools to 8x8, computes cross-attention in compact space, upsamples back
        # O(64^2) per head instead of O(4096^2) — CPU-efficient and novel
        self.caf = SpatialPooledCrossAttentionFusion(embed_dim=embed_dim, num_heads=4, pool_size=16, reduction=4)

        # ── Post-fusion refinement ──
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
        )

        # ── Pixel-shuffle upsampler ──
        if scale == 2:
            self.up = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            )
        elif scale == 3:
            self.up = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 9, 3, 1, 1),
                nn.PixelShuffle(3),
                nn.ReLU(inplace=True),
            )
        elif scale == 4:
            self.up = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unsupported scale factor: {scale}. Use 2, 3 or 4.")

        # ── Reconstruction head ──
        self.tail = nn.Conv2d(embed_dim, out_channels, 3, 1, 1)

        # ── Weight initialization ──
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize reconstruction tail to zero for stable residual learning
        nn.init.zeros_(self.tail.weight)
        if self.tail.bias is not None:
            nn.init.zeros_(self.tail.bias)

    def _encode(self, x):
        """Shared encoder: head + deep RCAB body + global residual."""
        feat = self.head(x)
        return feat + self.body_tail(self.body(feat))

    def forward(self, lr1, lr2):
        """
        Args:
            lr1: (B, 3, H, W) — first LR satellite capture
            lr2: (B, 3, H, W) — second LR satellite capture (slightly shifted)
        Returns:
            hr: (B, 3, H*scale, W*scale) — super-resolved HR output
        """
        # Independent deep feature extraction
        f1 = self._encode(lr1)
        f2 = self._encode(lr2)

        # Novel cross-attention fusion: aligns sub-pixel information
        fused = self.caf(f1, f2)

        # Post-fusion refinement
        fused = fused + self.refine(fused)  # local residual

        # Upsample to HR space
        res = self.tail(self.up(fused))

        # Global residual connection: bicubic upsample of lr1
        lr1_up = F.interpolate(lr1, scale_factor=self.scale, mode='bicubic', align_corners=False)

        return res + lr1_up

