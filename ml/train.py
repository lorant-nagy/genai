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


DEVICE = args.device

# Validate and set device
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    print(f"[bold red]CUDA not available, using CPU[/bold red]")
    # stop execution if cuda was explicitly requested
    sys.exit(1)
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

tags_from_config = getattr(wandb_config, 'tags', [])
if isinstance(tags_from_config, str):
    tags_from_config = [tags_from_config]

integrator_tag = config.corruption.integrator_cls
if hasattr(config.corruption.integrator_params, 'r'):
    integrator_tag = f"{integrator_tag}_r{config.corruption.integrator_params.r}"

tags = [config.corruption.process_cls, config.model.cls, integrator_tag] + tags_from_config

wandb.init(
        project=getattr(wandb_config, 'project', 'diffusion-training'),
        entity=getattr(wandb_config, 'entity', None),
        name=petname_str,
        config=config.to_dict(),
        dir=run_dir,
        mode=getattr(wandb_config, 'mode', 'online'),
        notes=getattr(wandb_config, 'notes', ''),
        tags=tags
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

# Store last 5 epochs for table display

# Print config once (using rich markup)
print(f"\n[bold green]{'='*105}[/bold green]")
print(f"[bold green]RUN:[/bold green] [bold yellow]{petname_str}[/bold yellow]  [dim]({time_str})[/dim]")
print(f"[bold green]{'='*105}[/bold green]\n")

print(f"[bold cyan]CONFIG:[/bold cyan]")
print(f"  [blue]Dataset:[/blue] [yellow]{dataset_name}[/yellow]  [blue]Model:[/blue] [yellow]{model_name}[/yellow]  [blue]Device:[/blue] [yellow]{DEVICE}[/yellow]")
print(f"  [blue]Process:[/blue] [yellow]{process_name}[/yellow]  [blue]Integrator:[/blue] [yellow]{integrator_name}[/yellow]  [blue]Loss:[/blue] [yellow]{config.loss.cls}[/yellow]")

# Get n_steps for main display
n_steps_display = "N/A"
if hasattr(config.corruption, 'corruptor_params') and hasattr(config.corruption.corruptor_params, 'n_steps'):
    n_steps_display = config.corruption.corruptor_params.n_steps

print(f"  [blue]Optimizer:[/blue] [yellow]{optimizer_name}[/yellow]  [blue]Epochs:[/blue] [yellow]{config.train.n_epochs}[/yellow]  [blue]Batch:[/blue] [yellow]{config.dataloader.batch_size}[/yellow]  [blue]Steps:[/blue] [yellow]{n_steps_display}[/yellow]")

# Model architecture details
if hasattr(config.model, 'model_params'):
    mp = config.model.model_params
    model_info = []
    if hasattr(mp, 'model_channels'):
        model_info.append(f"ch={mp.model_channels}")
    if hasattr(mp, 'channel_mult'):
        model_info.append(f"mult={mp.channel_mult}")
    if hasattr(mp, 'num_res_blocks'):
        model_info.append(f"res={mp.num_res_blocks}")
    if hasattr(mp, 'attention_resolutions'):
        model_info.append(f"attn={mp.attention_resolutions}")
    if hasattr(mp, 'dropout'):
        model_info.append(f"dropout={mp.dropout}")
    if model_info:
        print(f"  [dim]Model: {', '.join(model_info)}[/dim]")

# Optimizer parameters
if hasattr(config.train, 'optimizer_params'):
    op = config.train.optimizer_params
    opt_info = []
    if hasattr(op, 'lr'):
        opt_info.append(f"lr={op.lr}")
    if hasattr(op, 'weight_decay'):
        opt_info.append(f"wd={op.weight_decay}")
    if hasattr(op, 'betas'):
        opt_info.append(f"betas={op.betas}")
    if opt_info:
        print(f"  [dim]Opt: {', '.join(opt_info)}[/dim]")

# Process parameters
if hasattr(config.corruption, 'process_params'):
    pp = config.corruption.process_params
    proc_info = []
    if hasattr(pp, 'alpha'):
        proc_info.append(f"α={pp.alpha}")
    if hasattr(pp, 'T'):
        proc_info.append(f"T={pp.T}")
    if hasattr(pp, 'sigma'):
        proc_info.append(f"σ={pp.sigma}")
    if hasattr(pp, 't0'):
        proc_info.append(f"t0={pp.t0}")
    if hasattr(pp, 'c_alpha'):
        proc_info.append(f"c_α={pp.c_alpha}")
    if hasattr(pp, 'c_0'):
        proc_info.append(f"c0={pp.c_0}")
    if proc_info:
        print(f"  [dim]Process: {', '.join(proc_info)}[/dim]")

# Corruption/integrator details
corr_info = []
if hasattr(config.corruption, 'integrator_params') and hasattr(config.corruption.integrator_params, 'r'):
    corr_info.append(f"r={config.corruption.integrator_params.r}")
if corr_info:
    print(f"  [dim]Corruption: {', '.join(corr_info)}[/dim]")

# Dataloader info
dl_info = []
if hasattr(config.dataloader, 'num_workers'):
    dl_info.append(f"workers={config.dataloader.num_workers}")
if hasattr(config.dataloader, 'shuffle'):
    dl_info.append(f"shuffle={config.dataloader.shuffle}")
if dl_info:
    print(f"  [dim]Dataloader: {', '.join(dl_info)}[/dim]")

# Other info
if viz_every:
    print(f"  [dim]Viz: every {viz_every} epochs[/dim]")

if hasattr(config.dataset_train, 'normalizer_cls'):
    print(f"  [dim]Normalizer: {config.dataset_train.normalizer_cls}[/dim]")

print(f"\n[bold cyan]TRAINING:[/bold cyan]")
print(f"  [bold]{'Epoch':<10} {'Loss':<15} {'Status'}[/bold]")
print(f"  [dim]{'─'*10} {'─'*15} {'─'*10}[/dim]")

# Define custom WandB step metric for epoch-based logging
wandb.define_metric("epoch")
wandb.define_metric("epoch_loss", step_metric="epoch")
wandb.define_metric("generated_samples", step_metric="epoch")
wandb.define_metric("best_loss", step_metric="epoch")

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
    
    # FIX #4: loss_sum is already a Python float
    average_loss_per_sample = loss_sum / len(dataloader_train)
    
    # Check if we should generate samples
    should_visualize = viz_every is not None and viz_every > 0 and ((epoch + 1) % viz_every == 0 or (epoch + 1) == 1)
    
    # Store epoch data
    
    # Print simple one-line progress
    epoch_str = f"{epoch + 1}/{config.train.n_epochs}"
    loss_str = f"{average_loss_per_sample:.6f}"
    status_str = "[magenta bold]●[/magenta bold]" if should_visualize else ""
    print(f"  {epoch_str:<10} {loss_str:<15} {status_str}", flush=True)

    
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

torch.cuda.empty_cache()
del model

print("[bold green]---------------------------------------------------------------[/bold green]")
print(f"[bold green]RUN ENDED -- NAME : ------------------[/][yellow] {petname_str} [/][bold green]------------------[/]")
print("[bold green]---------------------------------------------------------------[/bold green]")