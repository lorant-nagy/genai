"""
Training visualization utilities for logging samples to wandb during training.
"""

import torch
import torchvision.utils
import wandb
from rich import print

from diff.sim_core import SDESolver
from diff.reverse import make_reverse
from diff.samplers import StationarySampler



def generate_samples(
    model,
    process,
    integrator,
    normalizer,
    config,
    dataset,
    device,
    n_samples=9,
    n_steps=None
):
    """
    Generate samples from the model and return as wandb.Image.
    
    Args:
        model: The score model
        process: The forward SDE process
        integrator: The integrator for reverse SDE
        normalizer: The normalizer for denormalization
        config: Config object
        dataset: Dataset (for C, H, W dimensions)
        device: Device to run on
        n_samples: Number of samples to generate
        n_steps: Number of reverse steps (fewer = faster)
        
    Returns:
        wandb.Image object or None if failed
    """
    model.eval()
    
    try:
        # Create reverse SDE
        CHW = (dataset.C, dataset.H, dataset.W)
        reverse_sde = make_reverse(
            process, 
            model, 
            config.corruption.process_params.T, 
            **config.reverse_params.to_dict(), 
            CHW=CHW
        )
        
        # Create solver
        solver = SDESolver(reverse_sde, integrator)
        
        # Sample from stationary distribution
        stationary_sampler = StationarySampler(config, device=device, equilibration_factor=5.0)
        x0 = stationary_sampler((n_samples, dataset.C, dataset.H, dataset.W))
        
        # Run reverse process (no seed = use PyTorch's default randomness)
        with torch.inference_mode():
            t_grid, X = solver.simulate(x0, n_steps=n_steps, seed=None)
        
        # Get final samples (latest time = reconstructed data)
        final_samples = X[-1]  # [n_samples, C, H, W]
        
        # Denormalize for visualization
        final_samples_vis = normalizer.denormalize(final_samples.cpu())
        final_samples_vis = final_samples_vis.clamp(0, 1)
        
        # Create grid
        nrow = int(n_samples ** 0.5)  # Square grid: 9->3x3, 16->4x4
        grid = torchvision.utils.make_grid(
            final_samples_vis, 
            nrow=nrow, 
            padding=2,
            pad_value=1.0  # White padding
        )
        
        # Convert to numpy for wandb (HWC format)
        grid_np = grid.permute(1, 2, 0).numpy()
        
        return wandb.Image(grid_np)
        
    except Exception as e:
        print(f"[red]Sample generation failed: {e}[/red]", flush=True)
        return None
    
    finally:
        model.train()


def get_data_samples(dataloader, normalizer, n_samples=9):
    """
    Get data samples as wandb.Image.
    
    Args:
        dataloader: Training dataloader
        normalizer: Normalizer for denormalization
        n_samples: Number of samples to log
        
    Returns:
        wandb.Image object or None if failed
    """
    try:
        sample_batch = next(iter(dataloader))[:n_samples]
        
        # Denormalize for visualization
        sample_batch_vis = normalizer.denormalize(sample_batch.cpu())
        sample_batch_vis = sample_batch_vis.clamp(0, 1)
        
        # Create grid
        nrow = int(n_samples ** 0.5)
        grid = torchvision.utils.make_grid(
            sample_batch_vis, 
            nrow=nrow, 
            padding=2, 
            pad_value=1.0
        )
        grid_np = grid.permute(1, 2, 0).numpy()
        
        return wandb.Image(grid_np, caption="Training data samples")
        
    except Exception as e:
        print(f"[red]Data sample loading failed: {e}[/red]")
        return None