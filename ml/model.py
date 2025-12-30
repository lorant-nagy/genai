import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.registry import register

# ============================================================================
# U-Net Architecture for Score-Based Diffusion
# ============================================================================

@register
class UNetScore(nn.Module):
    """
    U-Net architecture for score estimation in diffusion models.
    
    Modern U-Net with:
    - Multi-scale encoder-decoder structure
    - Skip connections between encoder and decoder
    - ResNet blocks with time conditioning
    - Self-attention at specified resolutions
    - GroupNorm for stability
    
    Args:
        in_channels: Input image channels (1 for grayscale, 3 for RGB)
        model_channels: Base channel count (scales with channel_mult)
        channel_mult: Channel multipliers per resolution [1, 2, 4] means 64, 128, 256
        num_res_blocks: Number of ResNet blocks per resolution level
        attention_resolutions: Resolutions to add self-attention [16, 8]
        dropout: Dropout probability
        time_dim: Time embedding dimension
    
    Example:
        >>> model = UNetScore(
        ...     in_channels=3,
        ...     model_channels=64,
        ...     channel_mult=[1, 2, 4],
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ...     dropout=0.1,
        ...     time_dim=128
        ... )
    """
    
    def __init__(
        self,
        in_channels: int,
        model_channels: int = 64,
        channel_mult: list = [1, 2, 4],
        num_res_blocks: int = 2,
        attention_resolutions: list = [16, 8],
        dropout: float = 0.0,
        time_dim: int = 128,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.attention_resolutions = attention_resolutions
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(embed_dim=time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4),
        )
        
        # Initial convolution
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        ])
        
        # Downsampling blocks
        input_block_channels = [model_channels]
        ch = model_channels
        ds = 1  # Current downsampling factor
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        channels=ch,
                        out_channels=mult * model_channels,
                        time_channels=time_dim * 4,
                        dropout=dropout
                    )
                ]
                ch = mult * model_channels
                
                # Add attention at specified resolutions
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_channels.append(ch)
            
            # Downsample (except at last level)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(ch))
                input_block_channels.append(ch)
                ds *= 2
        
        # Bottleneck
        self.middle_block = nn.Sequential(
            ResBlock(
                channels=ch,
                time_channels=time_dim * 4,
                dropout=dropout
            ),
            AttentionBlock(ch),
            ResBlock(
                channels=ch,
                time_channels=time_dim * 4,
                dropout=dropout
            ),
        )
        
        # Upsampling blocks
        self.output_blocks = nn.ModuleList([])
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # Pop skip connection channel count
                skip_ch = input_block_channels.pop()
                
                layers = [
                    ResBlock(
                        channels=ch + skip_ch,
                        out_channels=model_channels * mult,
                        time_channels=time_dim * 4,
                        dropout=dropout
                    )
                ]
                ch = model_channels * mult
                
                # Add attention at specified resolutions
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                
                # Upsample (except at first iteration of each level except first)
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                
                self.output_blocks.append(nn.Sequential(*layers))
        
        # Final output
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images
            t: (B,) time steps
        
        Returns:
            (B, C, H, W) score estimates
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (B, time_dim * 4)
        
        # Encoder (downsampling path)
        hs = []
        h = x
        for module in self.input_blocks:
            # Handle different module types
            if isinstance(module, nn.Conv2d):
                # Initial convolution
                h = module(h)
            elif isinstance(module, (Downsample, Upsample)):
                # Downsample/Upsample are single modules
                h = module(h)
            elif isinstance(module, nn.Sequential):
                # Sequential containing ResBlock and/or AttentionBlock
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
            else:
                # Fallback for other single modules
                h = module(h)
            
            hs.append(h)
        
        # Bottleneck
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Decoder (upsampling path)
        for module in self.output_blocks:
            # Concatenate skip connection
            h = torch.cat([h, hs.pop()], dim=1)
            
            # Handle layers in module
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
            else:
                # Single module (shouldn't happen, but safe)
                h = module(h)
        
        return self.out(h)


class ResBlock(nn.Module):
    """
    Residual block with time conditioning.
    
    Architecture:
        x → GroupNorm → SiLU → Conv3x3 → GroupNorm → (+ time_emb) → SiLU → Dropout → Conv3x3 → + x
    """
    
    def __init__(
        self,
        channels: int,
        out_channels: int = None,
        time_channels: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_channels = out_channels or channels
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
        )
        
        # Time conditioning (FiLM: scale and shift)
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels * 2),
        ) if time_channels else None
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        
        # Skip connection
        self.skip_connection = (
            nn.Conv2d(channels, out_channels, kernel_size=1)
            if channels != out_channels
            else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        h = self.in_layers(x)
        
        # Apply time conditioning
        if self.time_emb_proj is not None and t_emb is not None:
            t_emb = self.time_emb_proj(t_emb)  # (B, 2*C)
            scale, shift = t_emb.chunk(2, dim=1)  # (B, C), (B, C)
            scale = scale.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            h = h * (1 + scale) + shift
        
        h = self.out_layers(h)
        return h + self.skip_connection(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for capturing long-range dependencies.
    
    Uses multi-head self-attention with pre-normalization.
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Reshape to (B, C, H*W)
        h = h.reshape(B, C, H * W)
        
        # Compute Q, K, V
        qkv = self.qkv(h)  # (B, 3*C, H*W)
        q, k, v = qkv.chunk(3, dim=1)  # Each: (B, C, H*W)
        
        # Reshape for multi-head attention
        head_dim = C // self.num_heads
        q = q.reshape(B, self.num_heads, head_dim, H * W)
        k = k.reshape(B, self.num_heads, head_dim, H * W)
        v = v.reshape(B, self.num_heads, head_dim, H * W)
        
        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        scale = head_dim ** -0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * scale  # (B, num_heads, H*W, H*W)
        attn = F.softmax(attn, dim=-1)
        
        h = torch.einsum('bhnm,bhdm->bhdn', attn, v)  # (B, num_heads, head_dim, H*W)
        h = h.reshape(B, C, H * W)
        
        # Project out
        h = self.proj_out(h)
        h = h.reshape(B, C, H, W)
        
        return x + h


class Downsample(nn.Module):
    """Downsample by factor of 2 using stride-2 convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample by factor of 2 using nearest-neighbor + convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

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
        dtype = t.dtype
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), self.nf, device=device, dtype=dtype)
        )  # (nf,)
        angles = t[:, None] * freqs[None, :]               # (B, nf)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, 2*nf)
        return self.proj(emb)                               # (B, embed_dim)

# --- a tiny conv net with FiLM from time embedding ---
@register
class ScoreNet(nn.Module):
    """
    Input:  x  (B,C,H,W),  t (B,) (or reverse the flattened inside ... )
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