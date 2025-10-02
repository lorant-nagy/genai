import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.registry import register

# --- simple sinusoidal time encoding ---
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 64, n_frequencies: int = 16):
        super().__init__()
        self.nf = n_frequencies
        self.out_dim = 2 * n_frequencies
        # project sin/cos features to embed_dim
        self.proj = nn.Sequential(
            nn.Linear(self.out_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:   # t: (B,)
        B = t.shape[0]
        # frequencies: geometric progression like in diffusion/posenc
        device = t.device
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), self.nf, device=device)
        )  # (nf,)
        angles = t[:, None] * freqs[None, :]               # (B, nf)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, 2*nf)
        return self.proj(emb)                               # (B, embed_dim)

# --- a tiny conv net with FiLM from time embedding ---
@register
class ScoreNet(nn.Module):
    """
    Input:  x  (B,C,H,W),  t (B,)
    Output: score estimate (B,C,H,W)
    """
    def __init__(self, in_channels: int, hidden: int = 64, time_dim: int = 64):
        super().__init__()
        self.time_emb = TimeEmbedding(embed_dim=time_dim)

        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden, in_channels, kernel_size=3, padding=1)

        # FiLM (feature-wise linear modulation) from time embedding
        self.to_gamma1 = nn.Linear(time_dim, hidden)
        self.to_beta1  = nn.Linear(time_dim, hidden)
        self.to_gamma2 = nn.Linear(time_dim, hidden)
        self.to_beta2  = nn.Linear(time_dim, hidden)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)  -> time embedding
        temb = self.time_emb(t)  # (B, time_dim)

        h = self.conv1(x)        # (B, hidden, H, W)
        gamma1 = self.to_gamma1(temb).unsqueeze(-1).unsqueeze(-1)  # (B, hidden, 1, 1)
        beta1  = self.to_beta1(temb).unsqueeze(-1).unsqueeze(-1)   # (B, hidden, 1, 1)
        h = F.silu(h * (1 + gamma1) + beta1)

        h = self.conv2(h)
        gamma2 = self.to_gamma2(temb).unsqueeze(-1).unsqueeze(-1)
        beta2  = self.to_beta2(temb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h * (1 + gamma2) + beta2)

        out = self.conv3(h)      # (B, C, H, W)
        return out
