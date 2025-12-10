# samplers.py - Simple batch samplers for various distributions
import torch
import numpy as np
from typing import Optional, Union, Tuple
import random
import torch
from diff.sim_core import SDESolver
from utils.registry import REGISTRY
import copy

from utils.globals import DEVICE, DTYPE

class StationarySampler:
    """
    Sample from stationary distribution by running forward SDE from N(0,1).
    """
    
    def __init__(self, config, equilibration_factor: float = 5.0):
        """
        Args:
            equilibration_factor: Multiply T and n_steps by this factor
        """
        
        self.device = DEVICE
        self.equilibration_factor = equilibration_factor
        
        process_name = config.corruption.process_cls
        process_parameters = copy.deepcopy(config.corruption.process_params)
        process_parameters.T = process_parameters.T * equilibration_factor
        
        process_parameters.score_table_params = None  # No score table needed for forward process

        proc = REGISTRY[process_name](**process_parameters.to_dict())
        
        integrator_name = config.corruption.integrator_cls
        integrator_parameters = config.corruption.integrator_params
        integrator = REGISTRY[integrator_name](**integrator_parameters.to_dict())

        self.solver = SDESolver(proc, integrator)

        n_steps_base = config.corruption.corruptor_params.n_steps
        self.n_steps = int(n_steps_base * equilibration_factor)
    
    def __call__(self, shape: tuple) -> torch.Tensor:
        """
        Sample from stationary distribution.
        """
        # Start from N(0, 1)
        x0 = torch.randn(shape, device=self.device)
        with torch.no_grad():
            t_grid, X = self.solver.simulate(x0, n_steps=self.n_steps)
        return X[-1]


def standard_normal_BCHW(batch_size: int, channels: int, height: int, width: int, device: str = "cpu") -> torch.Tensor:
    return torch.randn((batch_size, channels, height, width), device=device, dtype=torch.get_default_dtype())

def standard_normal_flat(batch_size: int, channels: int, height: int, width: int, device: str = "cpu") -> torch.Tensor:
    return torch.randn((batch_size, channels * height * width), device=device, dtype=torch.get_default_dtype())

def generate_random_side_rectangles(dim = None, num_rectangles=1, revert_color = True, always_center = False, fixed_width_and_height_perc = None):
    """
    Generate random rectangles with constraints based on distance to nearest sides.
    
    Args:
        dim: Dimension of the output tensor (dim x dim)
        num_rectangles: Number of rectangles to generate (default: 1)
    
    Returns:
        torch.Tensor: (dim, dim) tensor with rectangles marked as 1, background as 0
    """
    # Initialize empty tensor
    tensor = torch.zeros(dim, dim)
    
    for _ in range(num_rectangles):
        # Step 1: Sample center point
        if always_center:
            center_x = int((dim - 1) / 2)
            center_y = int((dim - 1) / 2)
        else:
            center_x = random.uniform(0, dim - 1)
            center_y = random.uniform(0, dim - 1)
            
        # Step 2: Calculate distance to nearest sides
        dist_to_left = center_x
        dist_to_right = dim - 1 - center_x
        dist_to_top = center_y
        dist_to_bottom = dim - 1 - center_y
        
        # Find minimum distances for width and height constraints
        max_half_width = min(dist_to_left, dist_to_right)
        max_half_height = min(dist_to_top, dist_to_bottom)
        
        # Step 3: Sample a and b (width and height) with constraint < 2 * distance to nearest sides
        max_width = 2 * max_half_width
        max_height = 2 * max_half_height
        
        # Sample width and height (ensure they're at least 1 pixel)
        if fixed_width_and_height_perc is not None:
            a = fixed_width_and_height_perc * dim
            b = fixed_width_and_height_perc * dim
        else:
            a = random.uniform(1, max(1, max_width))
            b = random.uniform(1, max(1, max_height))
        
        # Step 4: Fit rectangle around center point
        half_a = a / 2
        half_b = b / 2
        
        # Calculate rectangle bounds
        left = max(0, int(center_x - half_a))
        right = min(dim - 1, int(center_x + half_a))
        top = max(0, int(center_y - half_b))
        bottom = min(dim - 1, int(center_y + half_b))
        
        # Fill the rectangle in the tensor
        tensor[top:bottom+1, left:right+1] = 1

        if revert_color:
            tensor = 1 - tensor  # Invert colors: rectangles become 0, background 1
    return tensor

def create_spiral_image(size=16, device="cpu"):
    """Create a black and white spiral image."""
    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(size, device=device), 
                         torch.arange(size, device=device), indexing='ij')
    
    # Center coordinates
    center = size // 2
    x_centered = x - center
    y_centered = y - center
    
    # Convert to polar coordinates
    r = torch.sqrt(x_centered**2 + y_centered**2)
    theta = torch.atan2(y_centered, x_centered)
    
    # Create spiral pattern (adjust parameters for different spiral shapes)
    spiral = torch.sin(2 * theta + 0.5 * r)
    
    # Convert to binary (black and white)
    spiral_binary = (spiral > 0).float()
    
    return spiral_binary

def create_three_circles_image(size=16, device="cpu", seed=None):
    """
    Create a black and white image with three random circles.
    
    Args:
        size: Size of the square image (size x size)
        device: PyTorch device
        seed: Random seed for reproducibility
        
    Returns:
        Binary image tensor of shape [size, size] with three circles
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(size, device=device, dtype=torch.float32), 
                         torch.arange(size, device=device, dtype=torch.float32), indexing='ij')
    
    # Initialize image as all black (0)
    image = torch.zeros(size, size, device=device, dtype=torch.float32)
    
    # Generate three random circles
    for i in range(3):
        # Random center coordinates (with some margin from edges)
        margin = size * 0.1  # 10% margin from edges
        center_x = torch.rand(1, device=device) * (size - 2*margin) + margin
        center_y = torch.rand(1, device=device) * (size - 2*margin) + margin
        
        # Random radius (between 10% and 40% of image size)
        min_radius = size * 0.1
        max_radius = size * 0.4
        radius = torch.rand(1, device=device) * (max_radius - min_radius) + min_radius
        
        # Create circle mask
        distances = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        circle_mask = distances <= radius
        
        # Add circle to image (union operation - any pixel in any circle is white)
        image = torch.maximum(image, circle_mask.float())
    
    return image


def gaussian_sampler(
    batch_size: int,
    dim: int = 1,
    mean: Union[float, torch.Tensor] = 0.0,
    std: Union[float, torch.Tensor] = 1.0,
    device: Optional[str] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Sample from a Gaussian distribution.
    
    Args:
        batch_size: Number of samples
        dim: Dimension of each sample
        mean: Mean (scalar or tensor of shape [dim])
        std: Standard deviation (scalar or tensor of shape [dim])
        device: Device to place samples on
        seed: Random seed
        
    Returns:
        Samples of shape [batch_size, dim] if dim > 1, or [batch_size] if dim == 1
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if dim == 1:
        # Return shape [batch_size] for 1D data
        samples = torch.randn(batch_size, device=device) * std + mean
    else:
        # Return shape [batch_size, dim] for multi-dimensional data
        samples = torch.randn(batch_size, dim, device=device)
        samples = samples * std + mean
    
    return samples


def bimodal_gaussian_sampler(
    batch_size: int,
    dim: int = 1,
    mean1: float = -10.0,
    std1: float = 0.5,
    mean2: float = 10.0,
    std2: float = 0.5,
    weight: float = 0.5,
    device: Optional[str] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Sample from a bimodal Gaussian distribution.
    Each sample is drawn from N(mean1, std1²) with probability weight,
    or from N(mean2, std2²) with probability 1-weight.
    
    Args:
        batch_size: Number of samples
        dim: Dimension of each sample
        mean1, std1: Parameters for first mode
        mean2, std2: Parameters for second mode
        weight: Probability of sampling from first mode
        device: Device to place samples on
        seed: Random seed
        
    Returns:
        Samples of shape [batch_size, dim] if dim > 1, or [batch_size] if dim == 1
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Decide which mode each sample comes from
    mode_selector = torch.rand(batch_size, device=device) < weight
    
    # Generate samples from both modes
    samples1 = torch.randn(batch_size, dim if dim > 1 else 1, device=device) * std1 + mean1
    samples2 = torch.randn(batch_size, dim if dim > 1 else 1, device=device) * std2 + mean2
    
    # Combine based on mode selection
    if dim == 1:
        samples1 = samples1.squeeze(-1)
        samples2 = samples2.squeeze(-1)
        samples = torch.where(mode_selector, samples1, samples2)
    else:
        mode_selector = mode_selector.unsqueeze(-1)  # [batch_size, 1]
        samples = torch.where(mode_selector, samples1, samples2)
    
    return samples


def uniform_box_sampler(
    batch_size: int,
    dim: int = 1,
    low: float = -0.5,
    high: float = 0.5,
    device: Optional[str] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Sample uniformly from a box [low, high]^dim.
    
    Args:
        batch_size: Number of samples
        dim: Dimension of each sample
        low: Lower bound of the box
        high: Upper bound of the box
        device: Device to place samples on
        seed: Random seed
        
    Returns:
        Samples of shape [batch_size, dim] if dim > 1, or [batch_size] if dim == 1
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if dim == 1:
        # Return shape [batch_size] for 1D data
        samples = torch.rand(batch_size, device=device) * (high - low) + low
    else:
        # Return shape [batch_size, dim] for multi-dimensional data
        samples = torch.rand(batch_size, dim, device=device) * (high - low) + low
    
    return samples


def image_noise_sampler(
    batch_size: int,
    channels: int = 3,
    height: int = 32,
    width: int = 32,
    mean: float = 0.0,
    std: float = 1.0,
    device: Optional[str] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Sample random noise images (e.g., for initializing diffusion models).
    
    Args:
        batch_size: Number of images
        channels: Number of channels (e.g., 3 for RGB)
        height: Image height
        width: Image width
        mean: Mean of the noise
        std: Standard deviation of the noise
        device: Device to place samples on
        seed: Random seed
        
    Returns:
        Noise images of shape [batch_size, channels, height, width]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    samples = torch.randn(batch_size, channels, height, width, device=device)
    samples = samples * std + mean
    
    return samples


# Convenience class-based samplers for compatibility
class GaussianSampler:
    """Class-based Gaussian sampler for repeated sampling with same parameters."""
    
    def __init__(
        self,
        dim: int = 1,
        mean: Union[float, torch.Tensor] = 0.0,
        std: Union[float, torch.Tensor] = 1.0,
        device: Optional[str] = None
    ):
        self.dim = dim
        self.mean = mean
        self.std = std
        self.device = device
    
    def __call__(self, batch_size: int, seed: Optional[int] = None) -> torch.Tensor:
        return gaussian_sampler(batch_size, self.dim, self.mean, self.std, self.device, seed)


class BimodalGaussianSampler:
    """Class-based bimodal Gaussian sampler for repeated sampling with same parameters."""
    
    def __init__(
        self,
        dim: int = 1,
        mean1: float = -10.0,
        std1: float = 0.5,
        mean2: float = 10.0,
        std2: float = 0.5,
        weight: float = 0.5,
        device: Optional[str] = None
    ):
        self.dim = dim
        self.mean1 = mean1
        self.std1 = std1
        self.mean2 = mean2
        self.std2 = std2
        self.weight = weight
        self.device = device
    
    def __call__(self, batch_size: int, seed: Optional[int] = None) -> torch.Tensor:
        return bimodal_gaussian_sampler(
            batch_size, self.dim, self.mean1, self.std1,
            self.mean2, self.std2, self.weight, self.device, seed
        )


class UniformBoxSampler:
    """Class-based uniform box sampler for repeated sampling with same parameters."""
    
    def __init__(
        self,
        dim: int = 1,
        low: float = -0.5,
        high: float = 0.5,
        device: Optional[str] = None
    ):
        self.dim = dim
        self.low = low
        self.high = high
        self.device = device
    
    def __call__(self, batch_size: int, seed: Optional[int] = None) -> torch.Tensor:
        return uniform_box_sampler(batch_size, self.dim, self.low, self.high, self.device, seed)