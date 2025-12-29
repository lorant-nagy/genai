"""
Dataset classes for training diffusion models.

All datasets follow the same interface:
- Return tensors of shape (C, H, W) in float32
- Have .C, .H, .W attributes for model configuration
- Accept a normalizer instance for data transformation
- Always generate on CPU for DataLoader compatibility
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.registry import register
from ml.normalizer import Normalizer
from diff.samplers import generate_random_side_rectangles


@register
class RectanglesDataset(Dataset):
    """
    Synthetic dataset generating random rectangles.
    
    Returns (C,H,W) with C=1 from a sampler that yields (H,W).
    Uses normalizer for data transformation.
    Always uses float32 for model compatibility.
    
    NOTE: Always generates tensors on CPU for DataLoader compatibility.
    Move to GPU in training loop with batch.to(device).
    
    Args:
        normalizer: Normalizer instance for data transformation
        C: Number of channels (always 1 for rectangles)
        H: Image height
        W: Image width
        always_center: If True, rectangles are always centered
        fixed_width_and_height_perc: If provided, use fixed size rectangles
        length: Number of samples in epoch (default: 2048)
    """
    def __init__(
        self,
        normalizer: Normalizer,
        C: int,
        H: int,
        W: int,
        always_center: bool = True,
        fixed_width_and_height_perc: Optional[float] = None,
        length: int = 128 * 16,
        **kwargs,
    ):
        self.normalizer = normalizer
        self.C = C
        self.H = H
        self.W = W
        # FIX #3: No device parameter - always generate on CPU
        self.sampler = generate_random_side_rectangles
        self.length = length
        self.ddim = H
        self.always_center = always_center
        self.fixed_width_and_height_perc = fixed_width_and_height_perc

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        # FIX #3: Always generate on CPU
        # Sampler creates tensors with default dtype (float32)
        # Returns values in {0, 1}
        img = self.sampler(
            self.ddim, 
            always_center=self.always_center, 
            fixed_width_and_height_perc=self.fixed_width_and_height_perc,
            device='cpu'  # Always CPU
        )
        img = img.unsqueeze(0)  # (H,W) -> (1,H,W)
        
        # Apply normalizer (replaces hardcoded 2.0 * img - 1.0)
        img = self.normalizer.normalize(img)
        
        return img


@register
class PokemonSpritesDataset(Dataset):
    """
    Dataset for Pokemon sprite images.
    
    Loads Pokemon sprites from a preprocessed directory created by
    tools/prepare_pokemon_sprites.py. Returns RGB images as (3,H,W) tensors.
    
    Args:
        normalizer: Normalizer instance for data transformation
        data_root: Path to preprocessed dataset (contains images/ and index.csv)
        image_size: Size to resize images to (height and width)
        max_items: Maximum number of images to load (None = all)
        pokedex_id: Filter to specific Pokemon by Pokedex ID (None = all)
        keep_shiny: Filter by shiny status (None=both, True=only shiny, False=only regular)
        interpolation: Resampling method ('nearest' for pixel art, 'bilinear' for smooth)
    
    Example:
        >>> normalizer = MinusOnePlusOneNormalizer()
        >>> dataset = PokemonSpritesDataset(
        ...     normalizer=normalizer,
        ...     data_root="/data/pokemon_sprites_parsed",
        ...     image_size=96
        ... )
        >>> img = dataset[0]  # Returns (3, 96, 96) tensor in normalized range
    """
    
    def __init__(
        self,
        normalizer: Normalizer,
        data_root: str,
        image_size: int = 96,
        max_items: Optional[int] = None,
        pokedex_id: Optional[int] = None,
        keep_shiny: Optional[bool] = None,
        interpolation: str = "nearest",
        **kwargs,
    ):
        self.normalizer = normalizer
        # FIX #3: No device parameter - datasets always work on CPU
        self.C = 3  # RGB
        self.H = image_size
        self.W = image_size
        
        # Setup resampling method
        if interpolation == "nearest":
            self._resample = Image.Resampling.NEAREST
        elif interpolation == "bilinear":
            self._resample = Image.Resampling.BILINEAR
        else:
            raise ValueError(
                f"interpolation must be 'nearest' or 'bilinear', got '{interpolation}'"
            )
        
        # Validate data_root
        root = Path(data_root).expanduser().resolve()
        img_dir = root / "images"
        if not img_dir.exists():
            raise FileNotFoundError(
                f"PokemonSpritesDataset: images directory not found: {img_dir}\n"
                f"Did you run tools/prepare_pokemon_sprites.py first?"
            )
        
        # Find all images
        exts = (".png", ".jpg", ".jpeg", ".webp")
        paths = []
        for p in img_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                # Skip __MACOSX files
                if "__MACOSX" not in str(p):
                    paths.append(p)
        
        if len(paths) == 0:
            raise RuntimeError(
                f"PokemonSpritesDataset: no images found in {img_dir}"
            )
        
        # Optional filtering by Pokedex ID (best-effort based on filename)
        if pokedex_id is not None:
            want = int(pokedex_id)
            filtered = []
            for p in paths:
                m = re.search(r"(?<!\d)(\d{1,4})(?!\d)", p.stem)
                if m and int(m.group(1)) == want:
                    filtered.append(p)
            
            if len(filtered) == 0:
                raise ValueError(
                    f"PokemonSpritesDataset: no images found for Pokedex ID {pokedex_id}"
                )
            
            paths = filtered
            print(f"Filtered to Pokedex ID {pokedex_id}: {len(paths)} images")
        
        # Optional filtering by shiny status (best-effort based on filename)
        if keep_shiny is not None:
            filtered = []
            for p in paths:
                s = str(p).lower()
                is_shiny = ("shiny" in s) or ("_shiny" in s) or ("-shiny" in s)
                if bool(is_shiny) == bool(keep_shiny):
                    filtered.append(p)
            
            if len(filtered) == 0:
                shiny_str = "shiny" if keep_shiny else "non-shiny"
                raise ValueError(
                    f"PokemonSpritesDataset: no {shiny_str} images found"
                )
            
            paths = filtered
            shiny_str = "shiny" if keep_shiny else "regular"
            print(f"Filtered to {shiny_str} sprites: {len(paths)} images")
        
        # Sort for deterministic ordering
        paths = sorted(paths)
        
        # Limit number of items
        if max_items is not None:
            paths = paths[:int(max_items)]
            print(f"Limited to {max_items} images")
        
        self.paths = paths
        
        print(f"PokemonSpritesDataset initialized:")
        print(f"  - Data root: {root}")
        print(f"  - Total images: {len(self.paths)}")
        print(f"  - Image size: {self.H}x{self.W}")
        print(f"  - Channels: {self.C} (RGB)")
        print(f"  - Interpolation: {interpolation}")
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and preprocess image.
        
        Returns:
            Tensor of shape (3, H, W) in normalized range (depends on normalizer)
            Always returns CPU tensor.
        """
        path = self.paths[idx]
        
        # Load image as RGB
        img = Image.open(path).convert("RGB")
        
        # Resize if needed
        if img.size != (self.W, self.H):
            img = img.resize((self.W, self.H), resample=self._resample)
        
        # Convert to numpy array in [0, 1] range
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
        
        # Convert to torch tensor (C, H, W) - on CPU
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        
        # Apply normalizer
        x = self.normalizer.normalize(x)
        
        return x.to(dtype=torch.float32)


@register
class OxfordFlowers102Dataset(Dataset):
    """
    Oxford Flowers 102 dataset.
    
    Loads flower images from a preprocessed directory created by
    tools/prepare_flowers.py. Returns RGB images as (3,H,W) tensors.
    
    Args:
        normalizer: Normalizer instance for data transformation
        data_root: Path to preprocessed dataset (contains images/)
        image_size: Size to resize images to (height and width)
        interpolation: Resampling method ('bilinear' for natural images, 'nearest' for pixel art)
    
    Example:
        >>> normalizer = MinusOnePlusOneNormalizer()
        >>> dataset = OxfordFlowers102Dataset(
        ...     normalizer=normalizer,
        ...     data_root="/data/flowers102",
        ...     image_size=64,
        ...     interpolation='bilinear'
        ... )
        >>> img = dataset[0]  # Returns (3, 64, 64) tensor
    """
    
    def __init__(
        self,
        normalizer: Normalizer,
        data_root: str,
        image_size: int = 64,
        interpolation: str = "bilinear",
        **kwargs,
    ):
        self.normalizer = normalizer
        self.C = 3  # RGB
        self.H = image_size
        self.W = image_size
        
        # Setup resampling method
        if interpolation == "nearest":
            self._resample = Image.Resampling.NEAREST
        elif interpolation == "bilinear":
            self._resample = Image.Resampling.BILINEAR
        elif interpolation == "bicubic":
            self._resample = Image.Resampling.BICUBIC
        else:
            raise ValueError(
                f"interpolation must be 'nearest', 'bilinear', or 'bicubic', got '{interpolation}'"
            )
        
        # Find all images in data_root
        root = Path(data_root).expanduser().resolve()
        
        # Find all image files recursively
        exts = (".png", ".jpg", ".jpeg", ".webp")
        paths = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                if "__MACOSX" not in str(p):
                    paths.append(p)
        
        # Sort for deterministic ordering
        self.paths = sorted(paths)
        
        print(f"OxfordFlowers102Dataset initialized:")
        print(f"  - Data root: {root}")
        print(f"  - Total images: {len(self.paths)}")
        print(f"  - Image size: {self.H}x{self.W}")
        print(f"  - Channels: {self.C} (RGB)")
        print(f"  - Interpolation: {interpolation}")
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and preprocess image.
        
        Returns:
            Tensor of shape (3, H, W) in normalized range (depends on normalizer)
            Always returns CPU tensor.
        """
        path = self.paths[idx]
        
        # Load image as RGB
        img = Image.open(path).convert("RGB")
        
        # Resize if needed
        if img.size != (self.W, self.H):
            img = img.resize((self.W, self.H), resample=self._resample)
        
        # Convert to numpy array in [0, 1] range
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
        
        # Convert to torch tensor (C, H, W) - on CPU
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        
        # Apply normalizer
        x = self.normalizer.normalize(x)
        
        return x.to(dtype=torch.float32)


@register
class StationaryDataset(Dataset):
    """
    Dataset that samples from the stationary distribution of a process.
    
    Uses StationarySampler internally. Instead of passing full config,
    we pass only the necessary process parameters.
    
    Args:
        normalizer: Normalizer instance for data transformation
        C: Number of channels
        H: Image height
        W: Image width
        device: Device to create tensors on
        length: Number of samples per epoch
        equilibration_factor: Multiply T by this for equilibration (default: 5.0)
        process_cls: Process class name (e.g., "SuperlinearLangevin")
        process_params: Dict of process parameters
        integrator_cls: Integrator class name (e.g., "EulerMaruyama")
        integrator_params: Dict of integrator parameters
        n_steps: Base number of steps for forward simulation
    """
    
    def __init__(
        self,
        normalizer: Normalizer,
        C: int,
        H: int,
        W: int,
        device: str = "cpu",
        length: int = 1000,
        equilibration_factor: float = 5.0,
        process_cls: str = "SuperlinearLangevin",
        process_params: dict = None,
        integrator_cls: str = "EulerMaruyama",
        integrator_params: dict = None,
        n_steps: int = 100,
        **kwargs,
    ):
        from diff.sim_core import SDESolver
        from utils.registry import REGISTRY
        
        self.normalizer = normalizer
        self.C = C
        self.H = H
        self.W = W
        self.device = device
        self.length = length
        
        # Setup process with longer T
        if process_params is None:
            raise ValueError("StationaryDataset requires process_params")
        
        # Copy and modify for equilibration
        proc_params = dict(process_params)
        T_base = proc_params.get('T', 1.0)
        proc_params['T'] = T_base * equilibration_factor
        proc_params['device'] = device
        
        # Don't need score table for forward sampling
        proc_params['score_table_params'] = None
        
        # Create process
        process = REGISTRY[process_cls](**proc_params)
        
        # Setup integrator
        if integrator_params is None:
            integrator_params = {}
        integrator = REGISTRY[integrator_cls](**integrator_params)
        
        # Create solver
        self.solver = SDESolver(process, integrator)
        self.n_steps = int(n_steps * equilibration_factor)
        
        print(f"StationaryDataset initialized:")
        print(f"  - Image size: {H}x{W}")
        print(f"  - Channels: {C}")
        print(f"  - Device: {device}")
        print(f"  - Process: {process_cls}")
        print(f"  - T (equilibrated): {proc_params['T']:.2f}")
        print(f"  - Steps: {self.n_steps}")
        print(f"  - Length: {length}")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Generate a stationary sample.
        
        Returns:
            Tensor of shape (C, H, W) in normalized range
        """
        # Start from N(0, 1)
        x0 = torch.randn(1, self.C, self.H, self.W, device=self.device)
        
        # Run forward to equilibrate
        with torch.no_grad():
            import random
            seed = random.randint(0, 2**32 - 1)
            t_grid, X = self.solver.simulate(x0, n_steps=self.n_steps, seed=seed)
        
        # Take final state
        sample = X[-1, 0]  # (C, H, W)
        
        # Apply normalizer
        sample = self.normalizer.normalize(sample)
        
        return sample.to(dtype=torch.float32)