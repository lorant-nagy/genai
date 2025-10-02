# rectangles_dataset.py
from dataclasses import dataclass
from typing import Callable, Optional, Dict
import torch
from torch.utils.data import Dataset
from utils.registry import register

from utils.samplers import generate_random_side_rectangles

@register
class RectanglesDataset(Dataset):
    """
    Returns (C,H,W) with C=1 from a sampler that yields (H,W).
    """
    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        always_center: bool = True,
        fixed_width_and_height_perc: Optional[float] = None,
        **kwargs,
    ):
        self.C = C
        self.H = H
        self.W = W
        self.sampler = generate_random_side_rectangles
        self.length = 128*16
        self.ddim = H
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.always_center = always_center
        self.fixed_width_and_height_perc = fixed_width_and_height_perc

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.sampler(self.ddim, always_center=self.always_center, fixed_width_and_height_perc=self.fixed_width_and_height_perc).to(self.device, dtype=torch.float32)
        img = img.to(dtype=self.dtype)
        img = img.unsqueeze(0)         # (H,W) -> (1,H,W)
        return img