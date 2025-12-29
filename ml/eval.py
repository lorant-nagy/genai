from torch.utils.data import DataLoader
import torch

# Force float32 as default dtype
torch.set_default_dtype(torch.float32)

import sys
import os
import wandb

sys.path.append(os.path.abspath(".."))
from utils.registry import REGISTRY
from utils.config import load_cfg
import argparse

from diff.sim_core import SDESolver
from diff.reverse import make_reverse
from diff.integrator import EulerMaruyama

from diff.samplers import StationarySampler
from utils.eval_plotting import plot_corruption_and_samples, plot_score_table_heatmaps, plot_density_table_heatmaps

import numpy as np

import matplotlib.pyplot as plt

from ml.dataset import RectanglesDataset
from ml.model import ScoreNet
from diff.sde import VPOU
from diff.corruptor import Corruptor
from diff.sim_core import SDESolver, ItoProcess

from rich import print


def eval(config, state_dict_path, eval_path, device="cpu"):
    print("*** starting evaluation", flush=True)
    
    DEVICE = device

    # Create normalizer first
    normalizer_name = config.dataset_eval.normalizer_cls
    normalizer_parameters = config.dataset_eval.normalizer_params
    normalizer = REGISTRY[normalizer_name](**normalizer_parameters.to_dict())

    # Create dataset with normalizer
    dataset_name = config.dataset_eval.dataset_cls
    dataset_parameters = config.dataset_eval.dataset_params
    dataset_parameters.device = DEVICE
    dataset_parameters.normalizer = normalizer

    # Special handling for StationaryDataset
    if dataset_name == "StationaryDataset":
        dataset_parameters.corruption_config = config.corruption
        # Don't convert to dict - pass object directly
        dataset = REGISTRY[dataset_name](
            **{k: v for k, v in dataset_parameters.__dict__.items() if k != 'corruption_config'},
            corruption_config=config.corruption
        )
    else:
        dataset = REGISTRY[dataset_name](**dataset_parameters.to_dict())

    # Fit normalizer if needed (should already be fitted from training, but just in case)
    if normalizer.needs_fitting:
        print(f"[bold yellow]Fitting normalizer to dataset...[/bold yellow]")
        normalizer.fit(dataset)
        print(f"[bold green]Normalizer fitted[/bold green]")

    # dataloader_parameters = config.dataset_eval.dataloader_params

    dataloader = DataLoader(dataset, **config.dataloader.to_dict())
    

    model_name = config.model.cls
    model_parameters = config.model.model_params
    model_parameters.in_channels = dataset.C
    model = REGISTRY[model_name](**model_parameters.to_dict())
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.to(device=DEVICE)
    model.eval()

    os.makedirs(eval_path, exist_ok=True)

    batch_of_images = next(iter(dataloader))
    batch_of_images = batch_of_images.to(device=DEVICE)

    process_name = config.corruption.process_cls
    process_parameters = config.corruption.process_params
    image_space_dim = dataset.C * dataset.H * dataset.W

    table_dir = os.path.join(config.env.results_dir, "score_tables")
    process_parameters.table_dir = table_dir

    proc = REGISTRY[process_name](**process_parameters.to_dict())

    integrator_name = config.corruption.integrator_cls
    integrator_parameters = config.corruption.integrator_params
    integrator = REGISTRY[integrator_name](**integrator_parameters.to_dict())

    corruptor_parameters = config.corruption.corruptor_params
    corruptor_parameters.integrator = integrator
    corruptor_parameters.process = proc
    # override mode : . \to "trajectory"
    corruptor_parameters.mode = "trajectory"
    corruptor = Corruptor(**corruptor_parameters.to_dict())

    CHW = (dataset.C, dataset.H, dataset.W)

    n_samples = config.eval.n_samples
    reverse_sde = make_reverse(proc, model, config.corruption.process_params.T, **config.reverse_params.to_dict(), CHW=CHW)
    integrator_name = config.corruption.integrator_cls
    integrator_parameters = config.corruption.integrator_params
    integrator = REGISTRY[integrator_name](**integrator_parameters.to_dict())
    backward_solver = SDESolver(reverse_sde, integrator)

    print(f"-------> creating stationary sampler", flush=True)
    stationary_sampler = StationarySampler(config, device=DEVICE, equilibration_factor=5.0)

    print(f"-------> sampling from stationary distribution", flush=True)
    x0 = stationary_sampler((n_samples, dataset.C, dataset.H, dataset.W))

    n_steps = config.corruption.corruptor_params.n_steps
    print(f"-------> simulating backward trajectories {n_samples} samples with {n_steps} steps", flush=True)
    with torch.inference_mode():
        t_grid, X = backward_solver.simulate(
                    x0,
                    n_steps=n_steps
                )

    print(f"-------> plotting", flush=True)
    
    # Get visualization range from normalizer
    vmin, vmax = normalizer.get_visualization_range()
    
    # Determine colormap (grayscale for C=1, none for RGB)
    cmap = "gray" if dataset.C == 1 else None

    # Plot corruption and samples (normalizer handles denormalization)
    plot_corruption_and_samples(
        corruptor=corruptor,
        batch_of_data=batch_of_images,  # Pass normalized data
        eval_path=eval_path,
        normalizer=normalizer,  # NEW: pass normalizer for denormalization
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        max_trajectories=5,
        sample_grid_count=9,
        show_time_header=True,
        mark_t0_indices=True,
        label_data_sample_indices=True,
        t_grid=t_grid,
        X=X[:, :9],  # Pass normalized data
        n_time_cols=10,
    )

    # final_samples = X_img_denorm[:, -1]  # Shape: (n_samples, C, H, W)
    # samples_np = final_samples.cpu().numpy()

    # Plot score table heatmaps if available
    if hasattr(proc, 'score_table_obj') and proc.score_table_obj is not None:
        print(f"-------> plotting score table heatmaps", flush=True)
        plot_score_table_heatmaps(
            score_table_obj=proc.score_table_obj,
            process=proc,
            eval_path=eval_path,
            n_time_points=10,
            dpi=150
        )
        
        # Also plot density table if available
        print(f"-------> plotting density table heatmaps", flush=True)
        plot_density_table_heatmaps(
            score_table_obj=proc.score_table_obj,
            process=proc,
            eval_path=eval_path,
            n_time_points=10,
            dpi=150,
            log_scale=False  # Linear scale works better for heatmaps (no striping)
        )
    else:
        print(f"-------> score table not available, skipping heatmaps", flush=True)

    # Log all plots to wandb (only if they exist)
    print(f"-------> logging to wandb", flush=True)
    
    plots_to_log = {
        "backward_final_samples": "backward_final_samples.png",
        "score_table_heatmaps": "score_table_heatmaps.png",
        "density_table_heatmaps": "density_table_heatmaps.png",
        "corruption_trajectories": "corruption_trajectories.png",
        "reverse_evolution": "reverse_evolution.png",
        "data_samples": "data_samples.png",
    }
    
    for key, filename in plots_to_log.items():
        filepath = os.path.join(eval_path, filename)
        if os.path.exists(filepath):
            wandb.log({key: wandb.Image(filepath)})
            print(f"  ✓ Logged {key}")
        else:
            print(f"  ✗ Skipped {key} (file not found)")
            
    
    if hasattr(config.eval, 'test') and config.eval.test:
        from utils.tests import test_stationary_recovery
        
        print(f"\n[bold cyan]-------> running stationary recovery test[/bold cyan]", flush=True)
        
        test_stationary_recovery(
            x0=x0,
            x_final=X[-1],
            config=config,
            save_dir=eval_path,
            device=DEVICE
        )

    wandb.run.log({}, commit=True)
    print("*** evaluation finished", flush=True)