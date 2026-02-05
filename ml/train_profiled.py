#train.py
import petname
import wandb
from rich import print
from rich.table import Table
from rich.console import Console
import numpy as np
import time
from collections import defaultdict

import argparse
import os
import sys
sys.path.append(os.path.abspath(".."))

import torch
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.float32)

from utils.viz import generate_samples
from utils.helpers import (
    init_wandb,
    maybe_update_best,
    collect_n_images,
    create_header,
    build_line,
    log_wandb_metrics,
    wandb_log_best_and_plot,
    log_wandb_best,
    fill_metrics_results,
    dump_dict_to_json,
    generate_samples_matplotlib
)
from utils.config import load_cfg
from utils.registry import REGISTRY

from ml.eval import compute_metrics

from diff.corruptor import Corruptor
from diff.sim_core import SDESolver

from diff.samplers import StationarySampler
from diff.reverse import make_reverse

# REGISTRY IMPORTS
import diff.sde
import diff.integrator
import ml.model
import ml.dataset
import ml.normalizer

from ml.eval import METRIC_KEYS

BENCHMARK_METRICS = ["loss", "fid", "kid_mean","w1_emb"]

# # # # # # # # PROFILING CLASS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class TrainingProfiler:
    """Tracks timing of different training phases"""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_times = {}
        self.epoch_start = None
        self.console = Console()
        
    def start(self, phase):
        """Start timing a phase"""
        self.start_times[phase] = time.perf_counter()
        
    def end(self, phase):
        """End timing a phase and record duration"""
        if phase in self.start_times:
            duration = time.perf_counter() - self.start_times[phase]
            self.timings[phase].append(duration)
            del self.start_times[phase]
            return duration
        return None
    
    def start_epoch(self):
        """Mark start of epoch"""
        self.epoch_start = time.perf_counter()
        
    def end_epoch(self):
        """Mark end of epoch and return duration"""
        if self.epoch_start is not None:
            duration = time.perf_counter() - self.epoch_start
            self.timings['total_epoch'].append(duration)
            self.epoch_start = None
            return duration
        return None
    
    def get_stats(self, phase):
        """Get statistics for a phase"""
        if phase not in self.timings or len(self.timings[phase]) == 0:
            return None
        times = self.timings[phase]
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'total': np.sum(times),
            'count': len(times)
        }
    
    def print_summary(self, epoch=None):
        """Print a summary table of all timings"""
        table = Table(title=f"Profiling Summary{f' (Epoch {epoch})' if epoch else ''}")
        table.add_column("Phase", style="cyan")
        table.add_column("Mean (s)", justify="right", style="green")
        table.add_column("Std (s)", justify="right")
        table.add_column("Min (s)", justify="right")
        table.add_column("Max (s)", justify="right")
        table.add_column("Total (s)", justify="right", style="yellow")
        table.add_column("Count", justify="right")
        table.add_column("% of Epoch", justify="right", style="magenta")
        
        # Get total epoch time for percentage calculation
        epoch_stats = self.get_stats('total_epoch')
        total_epoch_time = epoch_stats['mean'] if epoch_stats else 1.0
        
        # Sort phases by mean time (descending)
        phases = sorted(
            self.timings.keys(),
            key=lambda p: self.get_stats(p)['mean'] if self.get_stats(p) else 0,
            reverse=True
        )
        
        for phase in phases:
            stats = self.get_stats(phase)
            if stats:
                pct = (stats['mean'] / total_epoch_time * 100) if phase != 'total_epoch' else 100.0
                table.add_row(
                    phase,
                    f"{stats['mean']:.4f}",
                    f"{stats['std']:.4f}",
                    f"{stats['min']:.4f}",
                    f"{stats['max']:.4f}",
                    f"{stats['total']:.2f}",
                    f"{stats['count']}",
                    f"{pct:.1f}%"
                )
        
        self.console.print(table)
    
    def log_to_wandb(self, epoch):
        """Log timing statistics to wandb"""
        log_dict = {'epoch': epoch}
        for phase, times in self.timings.items():
            if len(times) > 0:
                log_dict[f'timing/{phase}_mean'] = np.mean(times)
                log_dict[f'timing/{phase}_last'] = times[-1]
        wandb.log(log_dict)

# # # # # # # # B L O C K 1  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("config", type=str, help="Path to the config file")
parser.add_argument("--device", type=int, default=0, help="CUDA device index (0,1,...)")
parser.add_argument("--profile-freq", type=int, default=10, help="How often to print profiling summary (epochs)")
args = parser.parse_args()

if not torch.cuda.is_available():
    print(f"[bold red]CUDA not available, using CPU[/bold red]")
    raise ValueError("CUDA not available on this machine.")
cuda_idx = args.device
torch.cuda.set_device(cuda_idx)
DEVICE = f"cuda:{cuda_idx}"
print(f"[bold blue]Dtype:[/] [yellow]{torch.get_default_dtype()}[/]")
print(f"[bold blue]Using device:[/] [yellow]{DEVICE}[/]")

config = load_cfg(args.config)
petname_str = petname.generate(2, separator="-")
time_str = os.popen("date +%Y%m%d_%H%M%S").read().strip()

adaptive_step = getattr(config.env, 'adaptive_step', False)
if adaptive_step:
    base = config.env.adaptive_base
    config.corruption.corruptor_params.n_steps = int(base * config.corruption.process_params.T)

metrics_freq = config.eval.metrics_freq
results_dir = config.env.results_dir
run_dir = os.path.join(results_dir, "runs", config.env.results_subdir, petname_str + "_" + time_str)
table_dir = os.path.join(results_dir, "score_tables")
os.makedirs(run_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)

# Initialize profiler
profiler = TrainingProfiler()

# # # # # # # # B L O C K 2 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# dump config
with open(os.path.join(run_dir, "config.yaml"), "w") as f:
    import yaml
    yaml.dump(config.to_dict(), f)

init_wandb(config, petname_str, run_dir)

profiler.start('setup_data')
normalizer_name = config.dataset_train.normalizer_cls
normalizer_parameters = config.dataset_train.normalizer_params
normalizer = REGISTRY[normalizer_name](**normalizer_parameters.to_dict())

dataset_name = config.dataset_train.dataset_cls
dataset_parameters = config.dataset_train.dataset_params
dataset_parameters.normalizer = normalizer
dataset = REGISTRY[dataset_name](**dataset_parameters.to_dict())
dataloader = DataLoader(dataset, **config.dataloader.to_dict())
profiler.end('setup_data')

profiler.start('setup_model')
model_name = config.model.cls
model_parameters = config.model.model_params
model_parameters.in_channels = dataset.C
model = REGISTRY[model_name](**model_parameters.to_dict())
model = model.to(DEVICE)
wandb.watch(model, log="all", log_freq=100)

optimizer_name = config.train.optimizer_cls
optimizer_parameters = config.train.optimizer_params
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_parameters.to_dict())
profiler.end('setup_model')

profiler.start('setup_sde')
process_name = config.corruption.process_cls
process_parameters = config.corruption.process_params
image_space_dim = dataset.C * dataset.H * dataset.W

process_parameters.table_dir = table_dir
process_parameters.device = DEVICE

proc = REGISTRY[process_name](**process_parameters.to_dict())

integrator_name = config.corruption.integrator_cls
integrator_parameters = config.corruption.integrator_params
integrator = REGISTRY[integrator_name](**integrator_parameters.to_dict())

corruptor_parameters = config.corruption.corruptor_params
corruptor_parameters.integrator = integrator
corruptor_parameters.process = proc
corruptor = Corruptor(**corruptor_parameters.to_dict())

loss_fn = getattr(torch.nn, config.loss.cls)(**config.loss.loss_params.to_dict())
stationary_sampler = StationarySampler(config, device=DEVICE, equilibration_factor=5.0)
profiler.end('setup_sde')

best_models = {key : {'value': float('inf'), 'state_dict': None, 'epoch': 0} for key in BENCHMARK_METRICS}
metrics_evo = {key : [] for key in ['epochs', 'loss', 'nan'] + METRIC_KEYS}

print("[bold green]---------------------------------------------------------------[/bold green]")
print(f"[bold green]RUN NAME : ------------------[/][yellow] {petname_str} [/][bold green]------------------[/bold green]")
print("[bold green]---------------------------------------------------------------[/bold green]")

print(f"\n[bold cyan]TRAINING:[/bold cyan]")

average_loss_per_sample = 0.0
loss_evo = []

profiler.start('collect_real_images')
real_norm = collect_n_images(dataloader, config.eval.n_metric_samples, device=DEVICE)
real_true = normalizer.denormalize(real_norm).clamp(0, 1)
profiler.end('collect_real_images')

print_tab_w = 16
header = create_header(metrics_evo, print_tab_w, external = ["nan%", "nan_step"])
print(header)

# # # # # # # # B L O C K 3 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

for epoch in range(config.train.n_epochs):
    profiler.start_epoch()
    
    profiler.start('training_loop')
    loss_sum = 0.0
    
    for c, batch in enumerate(dataloader):
        profiler.start('data_transfer')
        batch = batch.to(DEVICE)
        profiler.end('data_transfer')
        
        profiler.start('corruption')
        corrupted = corruptor(batch)
        profiler.end('corruption')
        
        profiler.start('score_computation')
        score = proc.score(corrupted['x'], batch, corrupted['t'])
        profiler.end('score_computation')
        
        profiler.start('forward_pass')
        prediction = model(corrupted['x'], corrupted['t'])
        profiler.end('forward_pass')
        
        profiler.start('loss_computation')
        output = loss_fn(prediction, score)
        loss_sum += output.item()
        profiler.end('loss_computation')
        
        profiler.start('backward_pass')
        output.backward()
        profiler.end('backward_pass')
        
        profiler.start('optimizer_step')
        optimizer.step()
        optimizer.zero_grad()
        profiler.end('optimizer_step')
    
    training_time = profiler.end('training_loop')
    
    average_loss_per_sample = loss_sum / len(dataloader)
    loss_evo.append(average_loss_per_sample)
    
    metrics_evo['epochs'].append(epoch+1)
    metrics_evo['loss'].append(loss_evo[-1])
    
    # # # E V A L U A T I O N # # #
    profiler.start('evaluation')
    
    profiler.start('eval_setup')
    model.eval()
    CHW = (dataset.C, dataset.H, dataset.W)
    reverse_sde = make_reverse(proc, model, config.corruption.process_params.T, **config.reverse_params.to_dict(), CHW=CHW)
    backward_solver = SDESolver(reverse_sde, integrator)

    x0 = stationary_sampler((config.eval.n_metric_samples, dataset.C, dataset.H, dataset.W))
    profiler.end('eval_setup')
    
    profiler.start('sampling')
    t_grid, X = backward_solver.simulate(
                x0,
                n_steps=config.corruption.corruptor_params.n_steps
            )
    profiler.end('sampling')

    profiler.start('nan_detection')
    gen_norm = X[-1].detach()
    
    total_steps = len(X)
    first_appearance = torch.full_like(gen_norm, total_steps, dtype=torch.float32)
    
    for step_idx, X_t in enumerate(X):
        nan_or_inf_mask = torch.isnan(X_t) | torch.isinf(X_t)
        newly_bad = nan_or_inf_mask & (first_appearance == total_steps)
        first_appearance[newly_bad] = step_idx
    
    affected_pixels = first_appearance < total_steps
    avg_first_appearance = first_appearance[affected_pixels].mean().item() if affected_pixels.any() else total_steps
    
    del X
    gen_true = normalizer.denormalize(gen_norm).clamp(0, 1)
    profiler.end('nan_detection')

    profiler.start('compute_metrics')
    metrics_results = compute_metrics(real_true, gen_true)
    fill_metrics_results(metrics_evo, metrics_results)
    profiler.end('compute_metrics')

    profiler.start('generate_samples_viz')
    sample_grid_matplotlib = generate_samples_matplotlib(
        model=model,
        process=proc,
        integrator=integrator,
        normalizer=normalizer,
        config=config,
        dataset=dataset,
        device=DEVICE,
        n_samples=8,
        n_steps=config.corruption.corruptor_params.n_steps
    )
    profiler.end('generate_samples_viz')
    
    model.train()
    
    eval_time = profiler.end('evaluation')
    # #e n d  o f  e v a l  s u b - b l o c k # # # # # # #

    profiler.start('logging')
    if sample_grid_matplotlib is not None:
        wandb.log({"generated_samples": sample_grid_matplotlib, "epoch": epoch + 1})

    for metric in BENCHMARK_METRICS:
        if metric in metrics_evo and len(metrics_evo[metric]) > 0:
            maybe_update_best(metric, metrics_evo[metric][-1], epoch+1, best_models, model)

    log_wandb_metrics(metrics_evo)
    profiler.end('logging')

    # NaN/Inf diagnostics
    n_images = gen_norm.shape[0]
    nan_or_inf_mask = torch.isnan(gen_norm) | torch.isinf(gen_norm)
    nan_or_inf_per_image = nan_or_inf_mask.view(n_images, -1).any(dim=1).float().mean().item() * 100
    
    external_dict = {
        "nan%": nan_or_inf_per_image,
        "nan_step": avg_first_appearance
    }

    epoch_time = profiler.end_epoch()
    
    line = build_line(metrics_evo, print_tab_w, external_dict=external_dict)
    print(line)
    
    # Print profiling summary periodically
    if (epoch + 1) % args.profile_freq == 0:
        print(f"\n[bold yellow]Profiling Summary after epoch {epoch+1}:[/bold yellow]")
        profiler.print_summary(epoch=epoch+1)
        profiler.log_to_wandb(epoch+1)
        print()

# # # # # # # # B L O C K 4 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

print("\n[bold yellow]Final Profiling Summary:[/bold yellow]")
profiler.print_summary()

log_wandb_best(best_models, config)

wandb_log_best_and_plot(
    model=model,
    proc=proc,
    config=config,
    integrator=integrator,
    stationary_sampler=stationary_sampler,
    normalizer=normalizer,
    best_models=best_models,
    C=dataset.C, H=dataset.H, W=dataset.W,
    epoch=config.train.n_epochs,
)

dump_dict_to_json(metrics_evo, dir=run_dir, filename="metrics_evo.json")

wandb.finish()
torch.cuda.empty_cache()
del model

print("[bold green]---------------------------------------------------------------[/bold green]")
print(f"[bold green]RUN ENDED -- NAME : ------------------[/][yellow] {petname_str} [/][bold green]------------------[/bold green]")
print("[bold green]---------------------------------------------------------------[/bold green]")