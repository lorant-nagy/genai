"""
Image Quality Metrics for Generative Models

This module provides standard metrics for evaluating generative models:
- FID (Fréchet Inception Distance)
- KID (Kernel Inception Distance)
- Wasserstein distances (W1, W2) in pixel and embedding space

All metrics use Inception V3 features for perceptual quality assessment.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, Union, Callable
from dataclasses import dataclass
import warnings

from rich import print


# ============================================================================
# MAIN API: compute_all_metrics
# ============================================================================

def compute_all_metrics(
    real_imgs: torch.Tensor,
    gen_imgs: torch.Tensor,
    device: str = 'cuda',
    batch_size: int = 50,
    kid_subset_size: int = 1000,
    kid_num_subsets: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Compute all image quality metrics at once.
    
    Computes:
    - FID (Fréchet Inception Distance)
    - KID (Kernel Inception Distance)
    - Wasserstein-1 distances (pixel and embedding space)
    - Wasserstein-2 distances (pixel and embedding space)
    
    Args:
        real_imgs: Real images, shape (N, C, H, W), normalized to [-1, 1] or [0, 1]
        gen_imgs: Generated images, shape (M, C, H, W), same normalization
        device: Device for computation ('cuda' or 'cpu')
        batch_size: Batch size for Inception forward passes
        kid_subset_size: Subset size for KID estimation
        kid_num_subsets: Number of subsets for KID variance
        verbose: Print progress messages
        
    Returns:
        Dictionary with all metrics:
        {
            'fid': float,                    # Fréchet Inception Distance
            'kid_mean': float,               # Kernel Inception Distance (mean)
            'kid_std': float,                # KID standard deviation
            'w1_pixel': float,               # Wasserstein-1 in pixel space
            'w2_pixel': float,               # Wasserstein-2 in pixel space
            'w1_embedding': float,           # Wasserstein-1 in embedding space
            'w2_embedding': float,           # Wasserstein-2 in embedding space
            'n_real': int,                   # Number of real samples
            'n_gen': int,                    # Number of generated samples
        }
    """
    
    if verbose:
        print(f"\n[bold cyan]Computing Image Quality Metrics[/bold cyan]")
        print(f"  Real samples: {real_imgs.shape}")
        print(f"  Generated samples: {gen_imgs.shape}")
        print(f"  Device: {device}")
    
    # Move to device
    real_imgs = real_imgs.to(device)
    gen_imgs = gen_imgs.to(device)
    
    # Safety check: Check for NaN/Inf
    if torch.isnan(real_imgs).any() or torch.isinf(real_imgs).any():
        if verbose:
            print(f"  [red]✗ Real images contain NaN/Inf[/red]")
        return {
            'fid': float('nan'), 'kid_mean': float('nan'), 'kid_std': float('nan'),
            'w1_pixel': float('nan'), 'w2_pixel': float('nan'),
            'w1_embedding': float('nan'), 'w2_embedding': float('nan'),
            'n_real': real_imgs.shape[0], 'n_gen': gen_imgs.shape[0]
        }
    
    if torch.isnan(gen_imgs).any() or torch.isinf(gen_imgs).any():
        if verbose:
            print(f"  [red]✗ Generated images contain NaN/Inf[/red]")
        return {
            'fid': float('nan'), 'kid_mean': float('nan'), 'kid_std': float('nan'),
            'w1_pixel': float('nan'), 'w2_pixel': float('nan'),
            'w1_embedding': float('nan'), 'w2_embedding': float('nan'),
            'n_real': real_imgs.shape[0], 'n_gen': gen_imgs.shape[0]
        }
    
    metrics = {
        'n_real': real_imgs.shape[0],
        'n_gen': gen_imgs.shape[0],
    }
    
    # -------------------------------------------------------------------------
    # Extract Inception Features (used for FID, KID, and W-embedding)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n  [yellow]Extracting Inception features...[/yellow]")
    
    # Initialize Inception model
    inception = InceptionV3Embeddings(device=device)
    
    # Extract features in batches
    real_features = inception.extract_features_batched(
        real_imgs, 
        batch_size=batch_size,
        verbose=verbose
    )
    gen_features = inception.extract_features_batched(
        gen_imgs, 
        batch_size=batch_size,
        verbose=verbose
    )
    
    if verbose:
        print(f"  [green]✓ Extracted features: {real_features.shape}[/green]")
    
    # -------------------------------------------------------------------------
    # Compute FID
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n  [yellow]Computing FID...[/yellow]")
    
    try:
        fid = calculate_fid(real_features, gen_features)
        metrics['fid'] = fid
        
        if verbose:
            print(f"  [green]✓ FID: {fid:.2f}[/green]")
    except Exception as e:
        print(f"  [red]✗ FID computation failed: {e}[/red]")
        metrics['fid'] = float('nan')
    
    # -------------------------------------------------------------------------
    # Compute KID
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n  [yellow]Computing KID...[/yellow]")
    
    try:
        kid_result = calculate_kid(
            real_features,
            gen_features,
            subset_size=kid_subset_size,
            num_subsets=kid_num_subsets,
            verbose=verbose
        )
        metrics['kid_mean'] = kid_result['kid_mean']
        metrics['kid_std'] = kid_result['kid_std']
        
        if verbose:
            print(f"  [green]✓ KID: {kid_result['kid_mean']:.6f} ± {kid_result['kid_std']:.6f}[/green]")
    except Exception as e:
        print(f"  [red]✗ KID computation failed: {e}[/red]")
        metrics['kid_mean'] = float('nan')
        metrics['kid_std'] = float('nan')
    
    # -------------------------------------------------------------------------
    # Compute Wasserstein Distances
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n  [yellow]Computing Wasserstein distances...[/yellow]")
    
    # Pixel space Wasserstein (W1 and W2)
    for p in [1, 2]:
        if verbose:
            print(f"    W{p} (pixel space)...", end=" ")
        
        try:
            w_pixel = wasserstein_distance_pot(
                real_imgs,
                gen_imgs,
                p=p,
                space='pixel',
                device=device,
                verbose=False
            )
            metrics[f'w{p}_pixel'] = w_pixel
            
            if verbose:
                print(f"[green]{w_pixel:.4f}[/green]")
        except Exception as e:
            print(f"[red]Failed: {e}[/red]")
            metrics[f'w{p}_pixel'] = float('nan')
    
    # Embedding space Wasserstein (W1 and W2)
    for p in [1, 2]:
        if verbose:
            print(f"    W{p} (embedding space)...", end=" ")
        
        try:
            w_embedding = wasserstein_distance_pot(
                real_features,
                gen_features,
                p=p,
                space='embedding',
                device=device,
                verbose=False
            )
            metrics[f'w{p}_embedding'] = w_embedding
            
            if verbose:
                print(f"[green]{w_embedding:.4f}[/green]")
        except Exception as e:
            print(f"[red]Failed: {e}[/red]")
            metrics[f'w{p}_embedding'] = float('nan')
    
    if verbose:
        print(f"\n[bold green]✓ Metrics computation complete[/bold green]\n")
    
    return metrics


# ============================================================================
# INCEPTION V3 EMBEDDINGS
# ============================================================================

class InceptionV3Embeddings:
    """
    Extract Inception V3 features for FID/KID computation.
    
    Uses the pool_3 layer (2048-dim) before final classification.
    Tries torch-fidelity first (standard), falls back to torchvision.
    
    Args:
        device: Device for computation
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = None
        self.backend = None
        
        # Try torch-fidelity first (standard for FID)
        try:
            from torch_fidelity.helpers import get_kwarg, vassert
            from torch_fidelity.feature_extractor_inceptionv3 import (
                FeatureExtractorInceptionV3
            )
            
            self.model = FeatureExtractorInceptionV3(
                name='inception-v3-compat',
                features_list=['2048']
            )
            self.model.eval()
            self.model.to(device)
            self.backend = 'torch_fidelity'
            
        except ImportError:
            warnings.warn(
                "torch-fidelity not available, falling back to torchvision. "
                "Install with: pip install torch-fidelity"
            )
        
        # Fallback to torchvision
        if self.model is None:
            import torchvision.models as models
            
            # Load pretrained Inception V3
            inception = models.inception_v3(pretrained=True, transform_input=False)
            inception.eval()
            
            # Remove final layers to get pool_3 features
            inception.fc = nn.Identity()
            inception.to(device)
            
            self.model = inception
            self.backend = 'torchvision'
    
    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for Inception V3.
        
        Handles both backends:
        - torch_fidelity: Requires uint8 tensors (0-255)
        - torchvision: Requires float32 tensors with ImageNet normalization
        
        Args:
            images: (N, C, H, W) in range [-1, 1] or [0, 1]
            
        Returns:
            Preprocessed images for Inception
        """
        # Ensure 3 channels (convert grayscale if needed)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Convert to [0, 1] range if in [-1, 1]
        if images.min() < 0:
            images = (images + 1.0) / 2.0
        
        # Clamp to [0, 1] to prevent numerical overshoots
        images = images.clamp(0, 1)
        
        # Resize to 299x299 (Inception input size)
        if images.shape[-1] != 299 or images.shape[-2] != 299:
            images = F.interpolate(
                images,
                size=(299, 299),
                mode='bilinear',
                align_corners=False
            )
        
        # Backend-specific preprocessing
        if self.backend == 'torch_fidelity':
            # torch_fidelity requires uint8 tensors (0-255)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            # torchvision requires float32 with ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.device)
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)
            images = (images - mean) / std
        
        return images
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a batch of images.
        
        Args:
            images: (N, C, H, W)
            
        Returns:
            features: (N, 2048)
        """
        images = self.preprocess_images(images)
        
        with torch.no_grad():
            if self.backend == 'torch_fidelity':
                result = self.model(images)
                
                # torch_fidelity can return different types:
                # - dict: {'2048': tensor}
                # - tuple: (tensor, ...) where first element is features
                # - tensor: features directly
                if isinstance(result, dict):
                    features = result['2048']
                elif isinstance(result, tuple):
                    features = result[0]  # First element is features
                else:
                    features = result  # Already a tensor
            else:  # torchvision
                features = self.model(images)
                
                # Ensure shape is (N, 2048)
                if features.dim() > 2:
                    features = F.adaptive_avg_pool2d(features, (1, 1))
                    features = features.squeeze(-1).squeeze(-1)
        
        return features
    
    def extract_features_batched(
        self,
        images: torch.Tensor,
        batch_size: int = 50,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Extract features in batches to avoid OOM.
        
        Args:
            images: (N, C, H, W)
            batch_size: Batch size for processing
            verbose: Print progress
            
        Returns:
            features: (N, 2048)
        """
        n_samples = images.shape[0]
        all_features = []
        
        for i in range(0, n_samples, batch_size):
            batch = images[i:i + batch_size]
            features = self.extract_features(batch)
            all_features.append(features.cpu())
            
            if verbose and ((i + batch_size) % 200 == 0 or (i + batch_size) >= n_samples):
                print(f"    Processed {min(i + batch_size, n_samples)}/{n_samples} images")
        
        return torch.cat(all_features, dim=0).to(self.device)


# ============================================================================
# FRÉCHET INCEPTION DISTANCE (FID)
# ============================================================================

def calculate_fid(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    eps: float = 1e-6
) -> float:
    """
    Calculate Fréchet Inception Distance (FID).
    
    FID measures the distance between two multivariate Gaussians:
        FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2√(Σ_r @ Σ_g))
    
    Lower FID = better quality and diversity.
    
    Args:
        real_features: (N, D) features from real images
        gen_features: (M, D) features from generated images
        eps: Small constant for numerical stability
        
    Returns:
        FID score (float)
    """
    # Compute mean and covariance
    mu_real = real_features.mean(dim=0)
    mu_gen = gen_features.mean(dim=0)
    
    sigma_real = torch_cov(real_features)
    sigma_gen = torch_cov(gen_features)
    
    # Compute mean difference
    diff = mu_real - mu_gen
    mean_dist = torch.sum(diff ** 2)
    
    # Compute trace term: Tr(Σ_r + Σ_g - 2√(Σ_r @ Σ_g))
    # Use scipy for matrix square root (more stable)
    covmean = sqrt_matrix_product(
        sigma_real.cpu().numpy(),
        sigma_gen.cpu().numpy(),
        eps=eps
    )
    covmean = torch.from_numpy(covmean).to(real_features.device)
    
    trace_term = torch.trace(sigma_real + sigma_gen - 2 * covmean)
    
    fid = mean_dist + trace_term
    
    return fid.item()


def torch_cov(X: torch.Tensor) -> torch.Tensor:
    """
    Compute covariance matrix.
    
    Args:
        X: (N, D) data matrix
        
    Returns:
        Cov: (D, D) covariance matrix
    """
    X_centered = X - X.mean(dim=0, keepdim=True)
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    return cov


def sqrt_matrix_product(A: np.ndarray, B: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute √(A @ B) using eigendecomposition.
    
    More numerically stable than direct square root.
    
    Args:
        A: (D, D) symmetric positive semi-definite matrix
        B: (D, D) symmetric positive semi-definite matrix
        eps: Small constant for numerical stability
        
    Returns:
        √(A @ B): (D, D) matrix
    """
    from scipy import linalg
    
    # Compute A @ B
    product = A @ B
    
    # Add small epsilon to diagonal for numerical stability
    product = product + eps * np.eye(product.shape[0])
    
    # Compute square root via eigendecomposition
    # For symmetric matrices: √M = Q √Λ Q^T
    eigvals, eigvecs = linalg.eigh(product)
    
    # Clip negative eigenvalues (numerical errors)
    eigvals = np.maximum(eigvals, 0)
    
    # Compute square root
    sqrt_eigvals = np.sqrt(eigvals)
    sqrt_product = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
    
    return sqrt_product


# ============================================================================
# KERNEL INCEPTION DISTANCE (KID)
# ============================================================================

def calculate_kid(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    kernel: str = 'polynomial',
    subset_size: int = 1000,
    num_subsets: int = 100,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef0: float = 1.0,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Calculate Kernel Inception Distance (KID).
    
    KID is an unbiased estimator of Maximum Mean Discrepancy (MMD):
        MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    
    where k is a kernel function (typically polynomial of degree 3).
    
    Uses subset sampling for efficiency and variance estimation.
    Lower KID = better match between distributions.
    
    Args:
        real_features: (N, D) features from real images
        gen_features: (M, D) features from generated images
        kernel: Kernel type ('polynomial' is standard for KID)
        subset_size: Size of random subsets for MMD estimation
        num_subsets: Number of subsets for mean/std computation
        degree: Polynomial kernel degree (default: 3)
        gamma: Kernel coefficient (default: 1/D)
        coef0: Kernel constant term (default: 1)
        verbose: Print progress
        
    Returns:
        {
            'kid_mean': Mean KID across subsets,
            'kid_std': Standard deviation of KID
        }
    """
    n_real = real_features.shape[0]
    n_gen = gen_features.shape[0]
    d = real_features.shape[1]
    
    # Set default gamma = 1/D (standard for KID)
    if gamma is None:
        gamma = 1.0 / d
    
    # Ensure we have enough samples
    subset_size = min(subset_size, n_real, n_gen)
    
    if verbose:
        print(f"    Using subsets of size {subset_size}, {num_subsets} subsets")
    
    # Compute KID for multiple random subsets
    kid_values = []
    
    for i in range(num_subsets):
        # Random subsets
        idx_real = torch.randperm(n_real)[:subset_size]
        idx_gen = torch.randperm(n_gen)[:subset_size]
        
        real_subset = real_features[idx_real]
        gen_subset = gen_features[idx_gen]
        
        # Compute MMD² for this subset
        mmd_sq = polynomial_mmd_squared(
            real_subset,
            gen_subset,
            degree=degree,
            gamma=gamma,
            coef0=coef0
        )
        
        kid_values.append(mmd_sq.item())
    
    kid_values = np.array(kid_values)
    
    return {
        'kid_mean': float(np.mean(kid_values)),
        'kid_std': float(np.std(kid_values))
    }


def polynomial_kernel(
    X: torch.Tensor,
    Y: torch.Tensor,
    degree: int = 3,
    gamma: float = 1.0,
    coef0: float = 1.0
) -> torch.Tensor:
    """
    Compute polynomial kernel matrix.
    
    K(x, y) = (gamma * <x, y> + coef0)^degree
    
    Args:
        X: (N, D) matrix
        Y: (M, D) matrix
        degree: Polynomial degree
        gamma: Scale factor
        coef0: Constant term
        
    Returns:
        K: (N, M) kernel matrix
    """
    # Compute dot products: X @ Y^T
    dot_products = X @ Y.T
    
    # Apply polynomial kernel
    K = (gamma * dot_products + coef0) ** degree
    
    return K


def polynomial_mmd_squared(
    X: torch.Tensor,
    Y: torch.Tensor,
    degree: int = 3,
    gamma: float = 1.0,
    coef0: float = 1.0
) -> torch.Tensor:
    """
    Compute unbiased estimator of squared MMD with polynomial kernel.
    
    MMD²(X, Y) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    
    Uses unbiased U-statistics (excludes diagonal).
    
    Args:
        X: (N, D) samples from distribution P
        Y: (M, D) samples from distribution Q
        degree, gamma, coef0: Polynomial kernel parameters
        
    Returns:
        Unbiased MMD² estimate (scalar tensor)
    """
    n = X.shape[0]
    m = Y.shape[0]
    
    # Compute kernel matrices
    K_XX = polynomial_kernel(X, X, degree, gamma, coef0)
    K_YY = polynomial_kernel(Y, Y, degree, gamma, coef0)
    K_XY = polynomial_kernel(X, Y, degree, gamma, coef0)
    
    # Unbiased estimator: exclude diagonal from K_XX and K_YY
    # E[k(x,x')] for x ≠ x'
    sum_XX = (K_XX.sum() - K_XX.trace()) / (n * (n - 1))
    sum_YY = (K_YY.sum() - K_YY.trace()) / (m * (m - 1))
    sum_XY = K_XY.sum() / (n * m)
    
    mmd_sq = sum_XX + sum_YY - 2 * sum_XY
    
    return mmd_sq


# ============================================================================
# WASSERSTEIN DISTANCE (via POT)
# ============================================================================

def wasserstein_distance_pot(
    samples_A: torch.Tensor,
    samples_B: torch.Tensor,
    p: int = 1,
    space: str = 'embedding',
    metric: str = 'euclidean',
    method: str = 'auto',
    device: str = 'cuda',
    reg: float = 0.01,
    numItermax: int = 100000,
    verbose: bool = False
) -> float:
    """
    Compute Wasserstein-p distance using Python Optimal Transport (POT).
    
    Supports both exact EMD and approximate Sinkhorn solvers.
    
    Args:
        samples_A: (N, D) samples from distribution P
        samples_B: (M, D) samples from distribution Q
        p: Order of Wasserstein distance (1 or 2)
        space: 'pixel' (flatten images) or 'embedding' (use as-is)
        metric: Distance metric for cost matrix ('euclidean', 'sqeuclidean')
        method: 'emd' (exact), 'sinkhorn' (approximate), 'auto' (choose based on size)
        device: Device for computation
        reg: Sinkhorn entropy regularization parameter
        numItermax: Max iterations for Sinkhorn
        verbose: Print progress
        
    Returns:
        Wasserstein distance (float)
    """
    try:
        import ot
    except ImportError:
        raise ImportError(
            "POT (Python Optimal Transport) not installed. "
            "Install with: pip install pot"
        )
    
    # Prepare samples
    if space == 'pixel':
        # Flatten images: (N, C, H, W) -> (N, C*H*W)
        samples_A = samples_A.reshape(samples_A.shape[0], -1)
        samples_B = samples_B.reshape(samples_B.shape[0], -1)
    
    # Convert to numpy for POT
    a_np = samples_A.cpu().numpy()
    b_np = samples_B.cpu().numpy()
    
    n = a_np.shape[0]
    m = b_np.shape[0]
    
    # Uniform weights
    weights_a = np.ones(n) / n
    weights_b = np.ones(m) / m
    
    # Compute cost matrix: C[i,j] = ||x_i - y_j||^p
    # For W_p, we need cost = distance^p
    if metric == 'euclidean':
        M = ot.dist(a_np, b_np, metric='euclidean')
    elif metric == 'sqeuclidean':
        M = ot.dist(a_np, b_np, metric='sqeuclidean')
    else:
        M = ot.dist(a_np, b_np, metric=metric)
    
    # Raise to power p if needed
    if p == 2 and metric == 'euclidean':
        M = M ** 2
    elif p != 1:
        M = M ** p
    
    # Choose solver
    if method == 'auto':
        # Use exact EMD for small problems, Sinkhorn for large
        method = 'emd' if (n * m < 1e6) else 'sinkhorn'
    
    if verbose:
        print(f"      Using {method} solver ({n} × {m} problem)")
    
    # Compute Wasserstein distance
    if method == 'emd':
        # Exact optimal transport
        try:
            wd = ot.emd2(weights_a, weights_b, M, numItermax=numItermax)
        except Exception as e:
            warnings.warn(f"EMD failed, falling back to Sinkhorn: {e}")
            wd = ot.sinkhorn2(weights_a, weights_b, M, reg=reg, numItermax=numItermax)
    else:
        # Sinkhorn (entropy-regularized)
        wd = ot.sinkhorn2(weights_a, weights_b, M, reg=reg, numItermax=numItermax)
    
    # Take p-th root to get W_p
    if p == 2:
        wd = np.sqrt(wd)
    
    return float(wd)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_samples_valid(samples: torch.Tensor, name: str = "samples") -> None:
    """Check if samples are valid (no NaN, Inf)."""
    if torch.isnan(samples).any():
        raise ValueError(f"{name} contain NaN values")
    if torch.isinf(samples).any():
        raise ValueError(f"{name} contain Inf values")


def print_metrics_summary(metrics: dict) -> None:
    """Pretty print metrics summary."""
    print(f"\n[bold cyan]Metrics Summary[/bold cyan]")
    print("=" * 50)
    
    if 'fid' in metrics:
        print(f"  FID:              {metrics['fid']:.2f}")
    
    if 'kid_mean' in metrics:
        print(f"  KID:              {metrics['kid_mean']:.6f} ± {metrics['kid_std']:.6f}")
    
    if 'w1_pixel' in metrics:
        print(f"  W1 (pixel):       {metrics['w1_pixel']:.4f}")
    
    if 'w2_pixel' in metrics:
        print(f"  W2 (pixel):       {metrics['w2_pixel']:.4f}")
    
    if 'w1_embedding' in metrics:
        print(f"  W1 (embedding):   {metrics['w1_embedding']:.4f}")
    
    if 'w2_embedding' in metrics:
        print(f"  W2 (embedding):   {metrics['w2_embedding']:.4f}")
    
    print("=" * 50 + "\n")


# ============================================================================
# HIGH-LEVEL: Complete Metrics Pipeline (Model + Dataset → Metrics)
# ============================================================================

def compute_metrics_with_model(
    config,
    model,
    dataset,
    dataloader,
    device: str = "cpu",
    log_prefix: str = "best_model",
    n_samples: Optional[int] = None,
    log_to_wandb: bool = True,
    verbose: bool = True,
    process = None,  # ← NEW: reuse existing process
    integrator = None  # ← NEW: reuse existing integrator
) -> Optional[dict]:
    """
    Complete metrics computation pipeline from model and dataset.
    
    This is the high-level function that:
    1. Collects real samples from dataloader
    2. Generates samples via reverse diffusion using the model
    3. Computes all metrics (FID, KID, W1, W2)
    4. Optionally logs to WandB with specified prefix
    
    Can be called from:
    - eval.py for full evaluation
    - train.py for periodic metrics during training
    - Any script that needs to evaluate a model
    
    Args:
        config: Configuration object
        model: Trained model (will be set to eval mode)
        dataset: Dataset for shape info
        dataloader: Dataloader for real samples
        device: Device for computation ('cuda' or 'cpu')
        log_prefix: WandB prefix ("best_model", "training", etc.)
        n_samples: Number of samples (None = use config.eval.n_metric_samples)
        log_to_wandb: Whether to log metrics to WandB (default: True)
        verbose: Whether to print progress messages (default: True)
        
    Returns:
        Dictionary of computed metrics, or None if computation fails
    """
    import os
    import torch
    import wandb
    from diff.reverse import make_reverse
    from diff.sim_core import SDESolver
    from diff.samplers import StationarySampler
    from utils.registry import REGISTRY
    
    if verbose:
        print(f"\n[bold cyan]-------> computing metrics ({log_prefix})[/bold cyan]", flush=True)
    
    # Get configuration
    if n_samples is None:
        n_samples = getattr(config.eval, 'n_metric_samples', 1000)
    
    metric_batch_size = getattr(config.eval, 'metric_batch_size', 50)
    kid_subset_size = getattr(config.eval, 'kid_subset_size', min(1000, n_samples))
    kid_num_subsets = getattr(config.eval, 'kid_num_subsets', 100)
    
    if verbose:
        print(f"  Collecting {n_samples} samples (batch size: {metric_batch_size})")
    
    # -------------------------------------------------------------------------
    # 1. COLLECT REAL SAMPLES from dataset
    # -------------------------------------------------------------------------
    if verbose:
        print(f"  [yellow]Collecting real samples...[/yellow]", flush=True)
    real_samples = []
    total_collected = 0
    
    for batch in dataloader:
        batch = batch.to(device=device)
        real_samples.append(batch)
        total_collected += batch.shape[0]
        
        if total_collected >= n_samples:
            break
    
    # Concatenate and trim to exact size
    real_samples = torch.cat(real_samples, dim=0)[:n_samples]
    if verbose:
        print(f"  [green]✓ Collected {real_samples.shape[0]} real samples[/green]")
    
    # -------------------------------------------------------------------------
    # 2. GENERATE SAMPLES from reverse diffusion
    # -------------------------------------------------------------------------
    if verbose:
        print(f"  [yellow]Generating samples...[/yellow]", flush=True)
    
    # Setup reverse process (reuse if provided, otherwise create)
    if process is None:
        process_name = config.corruption.process_cls
        process_parameters = config.corruption.process_params
        table_dir = os.path.join(config.env.results_dir, "score_tables")
        process_parameters.table_dir = table_dir
        proc = REGISTRY[process_name](**process_parameters.to_dict())
    else:
        proc = process
    
    if integrator is None:
        integrator_name = config.corruption.integrator_cls
        integrator_parameters = config.corruption.integrator_params
        integrator = REGISTRY[integrator_name](**integrator_parameters.to_dict())
    else:
        integrator = integrator
    
    CHW = (dataset.C, dataset.H, dataset.W)
    reverse_sde = make_reverse(
        proc, 
        model, 
        config.corruption.process_params.T,
        **config.reverse_params.to_dict(),
        CHW=CHW
    )
    backward_solver = SDESolver(reverse_sde, integrator)
    stationary_sampler = StationarySampler(config, device=device, equilibration_factor=5.0)
    
    n_steps = config.corruption.corruptor_params.n_steps
    
    # Generate in batches to avoid OOM
    generated_samples = []
    n_generated = 0
    
    model.eval()
    with torch.inference_mode():
        while n_generated < n_samples:
            current_batch_size = min(metric_batch_size, n_samples - n_generated)
            x0_batch = stationary_sampler((current_batch_size, dataset.C, dataset.H, dataset.W))
            t_grid_batch, X_batch = backward_solver.simulate(x0_batch, n_steps=n_steps)
            generated_samples.append(X_batch[-1])
            n_generated += current_batch_size
            
            if verbose and ((n_generated % 100 == 0) or (n_generated == n_samples)):
                print(f"    Generated {n_generated}/{n_samples}", flush=True)
    
    generated_samples = torch.cat(generated_samples, dim=0)
    if verbose:
        print(f"  [green]✓ Generated {generated_samples.shape[0]} samples[/green]")
    
    # -------------------------------------------------------------------------
    # 3. COMPUTE METRICS
    # -------------------------------------------------------------------------
    try:
        metrics = compute_all_metrics(
            real_imgs=real_samples,
            gen_imgs=generated_samples,
            device=device,
            batch_size=metric_batch_size,
            kid_subset_size=kid_subset_size,
            kid_num_subsets=kid_num_subsets,
            verbose=verbose
        )
        
        # Print summary (if verbose)
        if verbose:
            print_metrics_summary(metrics)
        
        # Log to WandB (optional)
        if log_to_wandb:
            wandb_metrics = {f'{log_prefix}/{k}': v for k, v in metrics.items()}
            wandb.log(wandb_metrics)
            if verbose:
                print(f"  [green]✓ Logged metrics to WandB ({log_prefix}/)[/green]")
        
        return metrics
        
    except ImportError as e:
        print(f"  [red]✗ Failed to import required dependencies: {e}[/red]")
        print(f"  [yellow]Install with: pip install pot torch-fidelity scipy>=1.7.0[/yellow]")
        return None
    except Exception as e:
        print(f"  [red]✗ Metrics computation failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None