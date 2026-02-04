#train.py
import petname
import wandb
from rich import print
import numpy as np

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
    dump_dict_to_json
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

# # # # # # # # B L O C K 1  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("config", type=str, help="Path to the config file")
parser.add_argument("--device", type=int, default=0, help="CUDA device index (0,1,...)")
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

metrics_freq = config.eval.metrics_freq
results_dir = config.env.results_dir
run_dir = os.path.join(results_dir, "runs", config.env.results_subdir, petname_str + "_" + time_str)
table_dir = os.path.join(results_dir, "score_tables")
os.makedirs(run_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)

# # # # # # # # B L O C K 2 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# dump config
with open(os.path.join(run_dir, "config.yaml"), "w") as f:
    import yaml
    yaml.dump(config.to_dict(), f)

init_wandb(config, petname_str, run_dir)

normalizer_name = config.dataset_train.normalizer_cls
normalizer_parameters = config.dataset_train.normalizer_params
normalizer = REGISTRY[normalizer_name](**normalizer_parameters.to_dict())

dataset_name = config.dataset_train.dataset_cls
dataset_parameters = config.dataset_train.dataset_params
dataset_parameters.normalizer = normalizer
dataset = REGISTRY[dataset_name](**dataset_parameters.to_dict())
dataloader = DataLoader(dataset, **config.dataloader.to_dict())

model_name = config.model.cls
model_parameters = config.model.model_params
model_parameters.in_channels = dataset.C
model = REGISTRY[model_name](**model_parameters.to_dict())
model = model.to(DEVICE)
wandb.watch(model, log="all", log_freq=100)

optimizer_name = config.train.optimizer_cls
optimizer_parameters = config.train.optimizer_params
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_parameters.to_dict())

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

best_models = {key : {'value': float('inf'), 'state_dict': None, 'epoch': 0} for key in BENCHMARK_METRICS}
metrics_evo = {key : [] for key in ['epochs', 'loss', 'nan'] + METRIC_KEYS}

print("[bold green]---------------------------------------------------------------[/bold green]")
print(f"[bold green]RUN NAME : ------------------[/][yellow] {petname_str} [/][bold green]------------------[/bold green]")
print("[bold green]---------------------------------------------------------------[/bold green]")

print(f"\n[bold cyan]TRAINING:[/bold cyan]")

average_loss_per_sample = 0.0
loss_evo = []

real_norm = collect_n_images(dataloader, config.eval.n_metric_samples, device=DEVICE)
real_true = normalizer.denormalize(real_norm).clamp(0, 1)

print_tab_w =16
header = create_header(metrics_evo, print_tab_w)
print(header)

# # # # # # # # B L O C K 3 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

for epoch in range(config.train.n_epochs):

    loss_sum = 0.0
    
    for c, batch in enumerate(dataloader):
        batch = batch.to(DEVICE)
        corrupted = corruptor(batch)
        score = proc.score(corrupted['x'], batch, corrupted['t'])
        prediction = model(corrupted['x'], corrupted['t'])
        output = loss_fn(prediction, score)
        
        loss_sum += output.item()
        
        output.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    average_loss_per_sample = loss_sum / len(dataloader)
    loss_evo.append(average_loss_per_sample)
    
    metrics_evo['epochs'].append(epoch+1)
    metrics_evo['loss'].append(loss_evo[-1])
    
    # # e v a l  s u b - b l o c k  W I T H  D I A G N O S T I C S # # # # # # #
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch+1} DIAGNOSTICS")
    print(f"{'='*80}")
    
    model.eval()
    CHW = (dataset.C, dataset.H, dataset.W)
    reverse_sde = make_reverse(proc, model, config.corruption.process_params.T, **config.reverse_params.to_dict(), CHW=CHW)
    backward_solver = SDESolver(reverse_sde, integrator)

    print(f"\n[1] INITIAL CONDITIONS (x0 from StationarySampler):")
    x0 = stationary_sampler((config.eval.n_metric_samples, dataset.C, dataset.H, dataset.W))
    print(f"  Shape: {x0.shape}, Device: {x0.device}")
    print(f"  Mean: {x0.mean():.6f}, Std: {x0.std():.6f}")
    print(f"  Min: {x0.min():.6f}, Max: {x0.max():.6f}")
    print(f"  NaN count: {torch.isnan(x0).sum().item()} / {x0.numel()}")
    print(f"  Inf count: {torch.isinf(x0).sum().item()} / {x0.numel()}")
    x0_finite = x0[~torch.isnan(x0) & ~torch.isinf(x0)]
    if len(x0_finite) > 0:
        print(f"  Finite values - Mean: {x0_finite.mean():.6f}, Std: {x0_finite.std():.6f}")
        print(f"  Finite values - Min: {x0_finite.min():.6f}, Max: {x0_finite.max():.6f}")

    print(f"\n[2] MODEL OUTPUT TEST (at t=0.5, t=3.5, t=6.5):")
    with torch.no_grad():
        for test_t_val in [0.5, 3.5, 6.5]:
            test_t = torch.full((min(10, x0.shape[0]),), test_t_val, device=DEVICE)
            test_x = x0[:min(10, x0.shape[0])]
            test_score = model(test_x, test_t)
            print(f"  t={test_t_val}:")
            print(f"    Mean: {test_score.mean():.6f}, Std: {test_score.std():.6f}")
            print(f"    Min: {test_score.min():.6f}, Max: {test_score.max():.6f}")
            print(f"    NaN count: {torch.isnan(test_score).sum().item()} / {test_score.numel()}")
            print(f"    Inf count: {torch.isinf(test_score).sum().item()} / {test_score.numel()}")
            score_finite = test_score[~torch.isnan(test_score) & ~torch.isinf(test_score)]
            if len(score_finite) > 0:
                print(f"    Finite - Mean: {score_finite.mean():.6f}, Std: {score_finite.std():.6f}")
                print(f"    Values > 100: {(torch.abs(score_finite) > 100).sum().item()}")
                print(f"    Values > 1000: {(torch.abs(score_finite) > 1000).sum().item()}")

    print(f"\n[3] BACKWARD SIMULATION:")
    t_grid, X = backward_solver.simulate(
                x0,
                n_steps=config.corruption.corruptor_params.n_steps
            )
    
    # Check trajectory at multiple time points
    check_indices = [0, len(X)//4, len(X)//2, 3*len(X)//4, -1]
    for i in check_indices:
        t_val = t_grid[i].item() if hasattr(t_grid[i], 'item') else t_grid[i]
        X_t = X[i]
        print(f"  Step {i}/{len(X)-1} (t={t_val:.3f}):")
        print(f"    Mean: {X_t.mean():.6f}, Std: {X_t.std():.6f}")
        print(f"    Min: {X_t.min():.6f}, Max: {X_t.max():.6f}")
        print(f"    NaN count: {torch.isnan(X_t).sum().item()} / {X_t.numel()}")
        print(f"    Inf count: {torch.isinf(X_t).sum().item()} / {X_t.numel()}")
        X_finite = X_t[~torch.isnan(X_t) & ~torch.isinf(X_t)]
        if len(X_finite) > 0:
            print(f"    Finite - Mean: {X_finite.mean():.6f}, Range: [{X_finite.min():.6f}, {X_finite.max():.6f}]")

    print(f"\n[4] FINAL GENERATED SAMPLES (X[-1]):")
    gen_norm = X[-1].detach()
    print(f"  Mean: {gen_norm.mean():.6f}, Std: {gen_norm.std():.6f}")
    print(f"  Min: {gen_norm.min():.6f}, Max: {gen_norm.max():.6f}")
    print(f"  NaN count: {torch.isnan(gen_norm).sum().item()} / {gen_norm.numel()}")
    print(f"  Inf count: {torch.isinf(gen_norm).sum().item()} / {gen_norm.numel()}")
    gen_finite = gen_norm[~torch.isnan(gen_norm) & ~torch.isinf(gen_norm)]
    if len(gen_finite) > 0:
        print(f"  Finite values - Mean: {gen_finite.mean():.6f}, Std: {gen_finite.std():.6f}")
        print(f"  Finite values - Range: [{gen_finite.min():.6f}, {gen_finite.max():.6f}]")
        print(f"  Values > 10: {(torch.abs(gen_finite) > 10).sum().item()}")
        print(f"  Values > 100: {(torch.abs(gen_finite) > 100).sum().item()}")
    
    del X
    gen_true = normalizer.denormalize(gen_norm).clamp(0, 1)
    
    print(f"\n[5] AFTER DENORMALIZATION:")
    print(f"  Mean: {gen_true.mean():.6f}, Std: {gen_true.std():.6f}")
    print(f"  Min: {gen_true.min():.6f}, Max: {gen_true.max():.6f}")
    print(f"  NaN count: {torch.isnan(gen_true).sum().item()} / {gen_true.numel()}")
    
    print(f"\n{'='*80}\n")

    # METRICS DISABLED - just fill with dummy values
    for key in METRIC_KEYS:
        metrics_evo[key].append(0.0)
    metrics_evo['nan'].append("")

    sample_grid = generate_samples(
        model=model,
        process=proc,
        integrator=integrator,
        normalizer=normalizer,
        config=config,
        dataset=dataset,
        device=DEVICE,
        n_samples=64,
        n_steps=config.corruption.corruptor_params.n_steps
    )
    
    model.train()
    # #e n d  o f  e v a l  s u b - b l o c k # # # # # # #

    if sample_grid is not None:
        wandb.log({"generated_samples": sample_grid, "epoch": epoch + 1})

    for metric in BENCHMARK_METRICS:
        if metric in metrics_evo and len(metrics_evo[metric]) > 0:
            maybe_update_best(metric, metrics_evo[metric][-1], epoch+1, best_models, model)

    log_wandb_metrics(metrics_evo)

    line = build_line(metrics_evo, print_tab_w)
    print(line)

# # # # # # # # B L O C K 4 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

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