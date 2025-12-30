import petname
import wandb

from rich import print

import argparse
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(".."))

import torch
# Force float32 as default dtype globally
torch.set_default_dtype(torch.float32)

from utils.config import load_cfg
from utils.registry import REGISTRY
from utils.viz import generate_samples
from utils.viz import get_data_samples

from ml.eval import eval

from torch.utils.data import DataLoader
from diff.corruptor import Corruptor
from diff.sim_core import SDESolver, ItoProcess

from ml.dataset import RectanglesDataset
from ml.model import ScoreNet
from diff.sde import VPOU
from diff.sde import SuperlinearLangevin


parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("config", type=str, help="Path to the config file")
parser.add_argument("--device", type=str, default=None, help="Device: 'cpu', 'cuda:0', 'cuda:1', etc.")
args = parser.parse_args()

config = load_cfg(args.config)

petname_str = petname.generate(2, separator="-")
time_str = os.popen("date +%Y%m%d_%H%M%S").read().strip()

print("[bold green]---------------------------------------------------------------[/bold green]")
print(f"[bold green]RUN NAME : ------------------[/][yellow] {petname_str} [/][bold green]------------------[/]")
print("[bold green]---------------------------------------------------------------[/bold green]")

# FIX #1: Device handling - read from env or command line
if args.device is not None:
    DEVICE = args.device
else:
    # Read from environment variable with fallback to auto-detect
    DEVICE = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

# Validate and set device
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    print(f"[bold red]CUDA not available, using CPU[/bold red]")
    DEVICE = "cpu"
elif DEVICE.startswith("cuda"):
    torch.cuda.set_device(DEVICE)
    print(f"[bold green]GPU: {torch.cuda.get_device_name(DEVICE)}[/bold green]")

print(f"[bold blue]Device:[/] [yellow]{DEVICE}[/]")
print(f"[bold blue]Dtype:[/] [yellow]float32[/]")

# Get visualization config
viz_every = getattr(config.train, 'visualize_every', None)
if viz_every is not None and viz_every > 0:
    print(f"[bold blue]Visualization:[/] [yellow]Every {viz_every} epochs[/]")
else:
    print(f"[bold blue]Visualization:[/] [yellow]Disabled[/]")

results_dir = config.env.results_dir
run_dir = os.path.join(results_dir, "runs", petname_str + "_" + time_str)
table_dir = os.path.join(results_dir, "score_tables")
os.makedirs(run_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)

with open(os.path.join(run_dir, "config.yaml"), "w") as f:
    import yaml
    yaml.dump(config.to_dict(), f)

# Initialize wandb
wandb_config = config.wandb
wandb.init(
        project=getattr(wandb_config, 'project', 'diffusion-training'),
        entity=getattr(wandb_config, 'entity', None),
        name=petname_str,
        config=config.to_dict(),
        dir=run_dir,
        mode=getattr(wandb_config, 'mode', 'online'),
        notes=getattr(wandb_config, 'notes', ''),
        tags=[config.corruption.process_cls, config.model.cls] + getattr(wandb_config, 'tags', [])
    )


# Create normalizer
normalizer_name = config.dataset_train.normalizer_cls
normalizer_parameters = config.dataset_train.normalizer_params
normalizer = REGISTRY[normalizer_name](**normalizer_parameters.to_dict())

# Create dataset 
dataset_name = config.dataset_train.dataset_cls
dataset_parameters = config.dataset_train.dataset_params
# FIX #3: Remove device parameter - datasets now always create on CPU
# dataset_parameters.device = DEVICE  # REMOVED
dataset_parameters.normalizer = normalizer  # Pass normalizer instance

if dataset_name == "StationaryDataset":
    dataset_parameters.corruption_config = config.corruption
    # Don't convert to dict - pass object directly
    dataset = REGISTRY[dataset_name](
        **{k: v for k, v in dataset_parameters.__dict__.items() if k != 'corruption_config'},
        corruption_config=config.corruption
    )
else:
    dataset = REGISTRY[dataset_name](**dataset_parameters.to_dict())

# Fit normalizer if needed (e.g., for data-dependent statistics)
if normalizer.needs_fitting:
    print(f"[bold yellow]Fitting normalizer to dataset...[/bold yellow]")
    normalizer.fit(dataset)
    print(f"[bold green]Normalizer fitted successfully[/bold green]")

dataloader_train = DataLoader(dataset, **config.dataloader.to_dict())

model_name = config.model.cls
model_parameters = config.model.model_params
model_parameters.in_channels = dataset.C

model = REGISTRY[model_name](**model_parameters.to_dict())
model = model.to(DEVICE)

# Watch model gradients
wandb.watch(model, log="all", log_freq=100)

optimizer_name = config.train.optimizer_cls
optimizer_parameters = config.train.optimizer_params
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_parameters.to_dict())

process_name = config.corruption.process_cls
process_parameters = config.corruption.process_params
image_space_dim = dataset.C * dataset.H * dataset.W

process_parameters.table_dir = table_dir
process_parameters.device = DEVICE

# Import the exception class to catch it
from diff.score_table import ScoreTableNumericalError

try:
    proc = REGISTRY[process_name](**process_parameters.to_dict())
except ScoreTableNumericalError as e:
    print(f"\n[bold red]Score table build failed: {str(e)}[/bold red]")
    wandb.log({"status": "terminated_score_table_error"})
    wandb.finish()
    sys.exit(1)

integrator_name = config.corruption.integrator_cls
integrator_parameters = config.corruption.integrator_params
integrator = REGISTRY[integrator_name](**integrator_parameters.to_dict())

corruptor_parameters = config.corruption.corruptor_params
corruptor_parameters.integrator = integrator
corruptor_parameters.process = proc
corruptor = Corruptor(**corruptor_parameters.to_dict())

loss_fn = getattr(torch.nn, config.loss.cls)(**config.loss.loss_params.to_dict())

# print info on training

print(f"-- dataset: {dataset_name}")
print(f"-- model: {model_name}")
print(f"-- corruption process: {process_name}")
print(f"-- corruption integrator: {integrator_name}")
print(f"-- optimizer: {optimizer_name}")
print(f"-- loss: {config.loss.cls}")
print(f"-- training for {config.train.n_epochs} epochs")

# Log initial data samples if visualization is enabled
if viz_every is not None and viz_every > 0:
    
    data_img = get_data_samples(dataloader_train, normalizer, n_samples=9)
    if data_img:
        wandb.log({"data_samples": data_img, "epoch": 0})
        print("[cyan]Initial data samples → logged[/cyan]")

print(f"[bold green]*** TRAINING STARTED[/bold green]")
best_model = None
best_loss = float('inf')
average_loss_per_sample = 0.0
loss_evo = []

for epoch in range(config.train.n_epochs):
    # FIX #4: Use Python float for loss accumulation
    loss_sum = 0.0  # Changed from loss = 0.0 (tensor accumulation)
    
    for c, batch in enumerate(dataloader_train):
        batch = batch.to(DEVICE)
        corrupted = corruptor(batch)
        score = proc.score(corrupted['x'], batch, corrupted['t'])
        prediction = model(corrupted['x'], corrupted['t'])
        output = loss_fn(prediction, score)
        
        # FIX #4: Accumulate scalar, not tensor
        loss_sum += output.item()  # Extract scalar immediately
        
        output.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Log to wandb
        wandb.log({
            "batch_loss": output.item()
        })
        
        print(f"batch {c+1}/{len(dataloader_train)}", end="\r", flush=True)
    
    # FIX #4: loss_sum is already a Python float
    average_loss_per_sample = loss_sum / len(dataloader_train)
    
    # Build epoch message
    epoch_msg = f"Epoch {epoch+1}/{config.train.n_epochs} || Loss: {average_loss_per_sample:.6f}"
    
    # Check if we should generate samples
    should_visualize = viz_every is not None and viz_every > 0 and ((epoch + 1) % viz_every == 0 or (epoch + 1) == 1)
    
    if should_visualize:
        epoch_msg += " → [magenta]generating samples...[/magenta]"
    
    print(epoch_msg, flush=True)
    
    loss_evo.append(average_loss_per_sample)
    
    # Log to wandb
    log_dict = {
        "epoch": epoch + 1,
        "epoch_loss": average_loss_per_sample,
        "best_loss": best_loss
    }
    
    # Generate and add samples if needed
    if should_visualize:
        sample_img = generate_samples(
            model=model,
            process=proc,
            integrator=integrator,
            normalizer=normalizer,
            config=config,
            dataset=dataset,
            device=DEVICE,
            n_samples=9,
            n_steps=config.corruption.corruptor_params.n_steps  # Quick sampling for speed
        )
        if sample_img:
            log_dict["generated_samples"] = sample_img
        else:
            print(f"                                → [red]sample generation failed[/red]", flush=True)
    
    wandb.log(log_dict)
    
    # Check if this is the best model (handles NaN/Inf properly)
    if not (torch.isnan(torch.tensor(average_loss_per_sample)) or 
            torch.isinf(torch.tensor(average_loss_per_sample))):
        if average_loss_per_sample < best_loss:
            best_loss = average_loss_per_sample
            best_model = model.state_dict()

# Safety: if no best model was saved (all losses were NaN/Inf), save the last model
if best_model is None:
    print("[bold yellow]Warning: No valid loss found during training. Saving last model state.[/bold yellow]")
    best_model = model.state_dict()
    best_loss = average_loss_per_sample

print(f"*** training finished", flush=True)

# save model
model_path = os.path.join(run_dir, "best_model.pth")
torch.save(best_model, model_path)
wandb.save(model_path)

# report training
print("[bold green]REPORT ON TRAINING:[/bold green]")
print(f"Best Loss: {best_loss:.6f}", flush=True)

# Log final summary
wandb.summary["best_loss"] = best_loss
wandb.summary["total_epochs"] = config.train.n_epochs

# evaluate
print("[bold green]*** STARTING EVALUATION[/bold green]")
eval(config, model_path, run_dir, device=DEVICE)

# plots
plt.figure()
plt.plot(loss_evo)
plt.xlabel("Epoch")
plt.ylabel("Average Loss per Sample")
plt.title("Loss Evolution During Training")
plt.grid()
loss_plot_path = os.path.join(run_dir, "loss_evolution.png")
plt.savefig(loss_plot_path)
plt.close()

# Log to wandb
wandb.log({"loss_evolution": wandb.Image(loss_plot_path)})



# Finish wandb
wandb.finish()

print("[bold green]---------------------------------------------------------------[/bold green]")
print(f"[bold green]RUN ENDED -- NAME : ------------------[/][yellow] {petname_str} [/][bold green]------------------[/]")
print("[bold green]---------------------------------------------------------------[/bold green]")