"""
Normalization/denormalization system for data preprocessing.

Normalizers transform data from its natural range (e.g., [0,1] for images)
to the range expected by the model (e.g., [-1,1]). They also provide
denormalization for visualization.

All normalizers are registry-based for config-driven instantiation.
"""

from __future__ import annotations
from typing import Tuple, Optional, Union
import torch
import numpy as np
from utils.registry import register


class Normalizer:
    """
    Base class for all normalizers.
    
    Normalizers transform data between two spaces:
    - Data space: natural range of the data (e.g., [0,1] for images)
    - Model space: range expected by the model (e.g., [-1,1])
    """
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform from data space to model space.
        
        Args:
            x: Input tensor in data space
            
        Returns:
            Tensor in model space
        """
        raise NotImplementedError
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform from model space back to data space.
        
        Args:
            x: Input tensor in model space
            
        Returns:
            Tensor in data space
        """
        raise NotImplementedError
    
    def get_visualization_range(self) -> Tuple[float, float]:
        """
        Get the (vmin, vmax) range for visualization after denormalization.
        
        Returns:
            Tuple of (vmin, vmax) for matplotlib imshow
        """
        raise NotImplementedError
    
    @property
    def needs_fitting(self) -> bool:
        """
        Whether this normalizer needs to be fitted to data before use.
        
        Returns:
            True if fit() must be called before normalize()
        """
        return False
    
    def fit(self, dataset) -> None:
        """
        Fit normalizer parameters to dataset (e.g., compute mean/std).
        
        Args:
            dataset: Dataset to compute statistics from
        """
        raise NotImplementedError("This normalizer does not support fitting")


@register
class MinusOnePlusOneNormalizer(Normalizer):
    """
    Maps [0, 1] → [-1, 1] for model input.
    
    This is the default normalization used in many diffusion models.
    Assumes input data is already in [0, 1] range (e.g., images normalized by 255).
    
    Formula:
        normalize: x → 2x - 1
        denormalize: x → (x + 1) / 2
    
    Example:
        >>> norm = MinusOnePlusOneNormalizer()
        >>> x = torch.tensor([0.0, 0.5, 1.0])
        >>> norm.normalize(x)
        tensor([-1.,  0.,  1.])
        >>> norm.denormalize(norm.normalize(x))
        tensor([0.0, 0.5, 1.0])
    """
    
    def __init__(self):
        super().__init__()
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map [0, 1] → [-1, 1]."""
        return 2.0 * x - 1.0
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map [-1, 1] → [0, 1]."""
        return (x + 1.0) / 2.0
    
    def get_visualization_range(self) -> Tuple[float, float]:
        """Return (0.0, 1.0) since denormalized data is in [0, 1]."""
        return (0.0, 1.0)


@register
class ZeroOneNormalizer(Normalizer):
    """
    Identity normalizer - keeps data in [0, 1] range.
    
    Useful when the model expects input in [0, 1] rather than [-1, 1].
    This is a no-op normalizer but provides consistent interface.
    
    Formula:
        normalize: x → x
        denormalize: x → x
    
    Example:
        >>> norm = ZeroOneNormalizer()
        >>> x = torch.tensor([0.0, 0.5, 1.0])
        >>> norm.normalize(x)
        tensor([0.0, 0.5, 1.0])
    """
    
    def __init__(self):
        super().__init__()
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Identity: x → x."""
        return x
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Identity: x → x."""
        return x
    
    def get_visualization_range(self) -> Tuple[float, float]:
        """Return (0.0, 1.0) since data stays in [0, 1]."""
        return (0.0, 1.0)


@register
class StandardNormalizer(Normalizer):
    """
    Z-score normalization using mean and standard deviation.
    
    Useful for datasets where pixel values should be centered and scaled,
    similar to ImageNet preprocessing.
    
    Formula:
        normalize: x → (x - mean) / std
        denormalize: x → x * std + mean
    
    Args:
        mean: Mean for normalization. Can be:
              - Single float (same for all channels)
              - List of floats (per-channel, length must match channels)
        std: Standard deviation for normalization. Same format as mean.
        data_range: Original data range (min, max) for visualization.
                   Default is (0.0, 1.0) for images.
    
    Example:
        >>> # ImageNet normalization
        >>> norm = StandardNormalizer(
        ...     mean=[0.485, 0.456, 0.406],
        ...     std=[0.229, 0.224, 0.225]
        ... )
        >>> x = torch.rand(3, 224, 224)  # RGB image in [0,1]
        >>> normalized = norm.normalize(x)
    """
    
    def __init__(
        self,
        mean: Union[float, list[float]],
        std: Union[float, list[float]],
        data_range: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        
        # Convert to tensors for efficient computation
        if isinstance(mean, (int, float)):
            self.mean = torch.tensor([mean], dtype=torch.float32)
        else:
            self.mean = torch.tensor(mean, dtype=torch.float32)
        
        if isinstance(std, (int, float)):
            self.std = torch.tensor([std], dtype=torch.float32)
        else:
            self.std = torch.tensor(std, dtype=torch.float32)
        
        # Store data range for visualization
        self.data_range = data_range
        
        # Validate
        if self.mean.shape != self.std.shape:
            raise ValueError(
                f"mean and std must have same shape, got {self.mean.shape} and {self.std.shape}"
            )
        
        if torch.any(self.std <= 0):
            raise ValueError("std must be positive")
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize: (x - mean) / std.
        
        Handles broadcasting for per-channel mean/std.
        """
        # Move mean/std to same device as input
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        
        # Reshape for broadcasting: (C,) → (C, 1, 1) for images
        if x.ndim > mean.ndim:
            shape = [mean.shape[0]] + [1] * (x.ndim - 1)
            mean = mean.reshape(shape)
            std = std.reshape(shape)
        
        return (x - mean) / std
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize: x * std + mean.
        """
        # Move mean/std to same device as input
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        
        # Reshape for broadcasting
        if x.ndim > mean.ndim:
            shape = [mean.shape[0]] + [1] * (x.ndim - 1)
            mean = mean.reshape(shape)
            std = std.reshape(shape)
        
        return x * std + mean
    
    def get_visualization_range(self) -> Tuple[float, float]:
        """
        Return original data range for visualization.
        
        Note: After denormalization, data should be clipped to this range.
        """
        return self.data_range


@register
class DataDependentStandardNormalizer(Normalizer):
    """
    Standard normalization with mean/std computed from dataset.
    
    Unlike StandardNormalizer, this computes mean and std by iterating
    over the dataset. Useful when you don't know the statistics ahead of time.
    
    This normalizer MUST be fitted before use by calling fit(dataset).
    
    Args:
        data_range: Original data range (min, max) for visualization.
                   Default is (0.0, 1.0) for images.
        max_samples: Maximum number of samples to use for computing statistics.
                     None means use entire dataset.
    
    Example:
        >>> norm = DataDependentStandardNormalizer()
        >>> norm.fit(dataset)  # Compute mean/std from data
        >>> x = dataset[0]
        >>> normalized = norm.normalize(x)
    """
    
    def __init__(
        self,
        data_range: Tuple[float, float] = (0.0, 1.0),
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_range = data_range
        self.max_samples = max_samples
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self._is_fitted = False
    
    @property
    def needs_fitting(self) -> bool:
        """This normalizer requires fitting."""
        return True
    
    def fit(self, dataset) -> None:
        """
        Compute mean and std from dataset.
        
        Args:
            dataset: Dataset to compute statistics from.
                    Should support indexing and len().
        """
        print(f"Computing normalization statistics from dataset...")
        
        # Determine number of samples to use
        n_total = len(dataset)
        n_samples = n_total if self.max_samples is None else min(self.max_samples, n_total)
        
        # Collect samples
        samples = []
        indices = np.linspace(0, n_total - 1, n_samples, dtype=int)
        
        for idx in indices:
            x = dataset[idx]
            if torch.is_tensor(x):
                samples.append(x.cpu())
            else:
                samples.append(torch.tensor(x))
        
        # Stack into single tensor: (N, C, H, W) or (N, C)
        samples = torch.stack(samples)
        
        # Compute per-channel statistics
        # samples shape: (N, C, ...) → compute over (N, ...) dimensions
        axes = tuple(range(1, samples.ndim))  # All axes except channel
        if samples.ndim > 1:
            axes = (0,) + tuple(range(2, samples.ndim))  # (N, spatial dims), keep C
        else:
            axes = (0,)  # Just batch dimension
        
        self.mean = samples.mean(dim=axes)
        self.std = samples.std(dim=axes)
        
        # Ensure std is not zero (add small epsilon)
        self.std = torch.clamp(self.std, min=1e-6)
        
        self._is_fitted = True
        
        print(f"  Computed statistics from {n_samples} samples")
        print(f"  Mean: {self.mean.tolist()}")
        print(f"  Std: {self.std.tolist()}")
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize using computed mean/std."""
        if not self._is_fitted:
            raise RuntimeError(
                "DataDependentStandardNormalizer must be fitted before use. "
                "Call normalizer.fit(dataset) first."
            )
        
        # Same logic as StandardNormalizer
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        
        if x.ndim > mean.ndim:
            shape = [mean.shape[0]] + [1] * (x.ndim - 1)
            mean = mean.reshape(shape)
            std = std.reshape(shape)
        
        return (x - mean) / std
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize using computed mean/std."""
        if not self._is_fitted:
            raise RuntimeError(
                "DataDependentStandardNormalizer must be fitted before use. "
                "Call normalizer.fit(dataset) first."
            )
        
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        
        if x.ndim > mean.ndim:
            shape = [mean.shape[0]] + [1] * (x.ndim - 1)
            mean = mean.reshape(shape)
            std = std.reshape(shape)
        
        return x * std + mean
    
    def get_visualization_range(self) -> Tuple[float, float]:
        """Return original data range for visualization."""
        return self.data_range