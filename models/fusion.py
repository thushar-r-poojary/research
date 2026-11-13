import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPooledCrossAttentionFusion(nn.Module):
    """
    Channel-wise Transposed Cross-Attention Fusion (TCAF) for Dual-Image SR.
    Computes cross-attention across channels rather than spatial locations to prevent
    information loss from spatial pooling, resolving blurriness while maintaining
    O(C^2) linear spatial complexity.
    """
    def __init__(self, embed_dim=64, num_heads=4, pool_size=16, reduction=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool_size = pool_size  # Keep for compatibility

        # Projections (1x1 convs)
        self.proj_q = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        self.proj_k = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        self.proj_v = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)

        # Output projection
        self.proj_out = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )

        # Adaptive gating
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // reduction, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        """
        Args:
          feat1: (B, C, H, W)
          feat2: (B, C, H, W)
        """
        B, C, H, W = feat1.shape
        N = H * W

        # Project features
        Q = self.proj_q(feat1).view(B, self.num_heads, self.head_dim, N)  # B, h, d, N
        K = self.proj_k(feat2).view(B, self.num_heads, self.head_dim, N)  # B, h, d, N
        V = self.proj_v(feat2).view(B, self.num_heads, self.head_dim, N)  # B, h, d, N

        # Normalize Q and K along spatial dimensions for training stability
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        # Transposed attention map (correlation along channels): (B, h, d, d)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        # Attend V
        out = torch.matmul(attn, V).contiguous().view(B, C, H, W)

        # Post-projection & channel gating
        out = self.proj_out(out)
        gate = self.gate(feat1 + out).view(B, C, 1, 1)

        return feat1 + gate * out
