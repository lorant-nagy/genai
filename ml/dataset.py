# rectangles_dataset.py
from dataclasses import dataclass
from typing import Callable, Optional, Dict
import torch
from torch.utils.data import Dataset
from utils.registry import register

from diff.samplers import generate_random_side_rectangles

@register
class RectanglesDataset(Dataset):
    """
    Returns (C,H,W) with C=1 from a sampler that yields (H,W).
    Data is normalized to [-1, 1] range.
    Always uses float32 for model compatibility.
    """
    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        always_center: bool = True,
        fixed_width_and_height_perc: Optional[float] = None,
        device: str = "cpu",  # Add device parameter with default
        **kwargs,
    ):
        self.C = C
        self.H = H
        self.W = W
        self.sampler = generate_random_side_rectangles
        self.length = 128*16
        self.ddim = H
        self.device = device  # Use parameter instead of global
        self.always_center = always_center
        self.fixed_width_and_height_perc = fixed_width_and_height_perc

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Sampler creates tensors with default dtype (float32)
        img = self.sampler(
            self.ddim, 
            always_center=self.always_center, 
            fixed_width_and_height_perc=self.fixed_width_and_height_perc,
            device=self.device
        )
        img = img.unsqueeze(0)  # (H,W) -> (1,H,W)
        
        # Sampler returns {0, 1}, we map to {-1, 1}
        img = 2.0 * img - 1.0
        
        return img