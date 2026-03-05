#helpers.py
from ml.eval import METRIC_KEYS
import wandb
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from diff.reverse import make_reverse
from diff.sim_core import SDESolver
import json
import os
from diff.samplers import StationarySampler
import yaml

class Cfg:
    def __init__(self, data):
        for k, v in (data or {}).items():
            if isinstance(v, dict):
                v = Cfg(v)
            elif isinstance(v, list):
                v = [Cfg(i) if isinstance(i, dict) else i for i in v]
            setattr(self, k, v)

    def to_dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Cfg):
                v = v.to_dict()
            elif isinstance(v, list):
                v = [i.to_dict() if isinstance(i, Cfg) else i for i in v]
            result[k] = v
        return result

def load_cfg(path: str) -> Cfg:
    with open(path, "r") as f:
        return Cfg(yaml.safe_load(f))

def generate_8x8_grid(model, proc, integrator, normalizer, config, C, H, W, device, seed=None):
    """
    Generate 64 samples and return as a numpy image (H_grid, W_grid, C) or (H_grid, W_grid) for grayscale.
    Model must already have the desired weights loaded and be in eval mode.
    """
    from diff.reverse import make_reverse
    from diff.sim_core import SDESolver
    from torchvision.utils import make_grid

    CHW = (C, H, W)
    reverse_sde = make_reverse(proc, model, config.corruption.process_params.T, **config.reverse_params.to_dict(), CHW=CHW)
    solver = SDESolver(reverse_sde, integrator)

    stationary_sampler = StationarySampler(config, device=device)
    x0 = stationary_sampler((64, C, H, W), seed=seed)

    with torch.no_grad():
        _, X = solver.simulate(x0, n_steps=config.corruption.corruptor_params.n_steps, seed=seed)
        gen_true = normalizer.denormalize(X[-1].detach()).clamp(0, 1)

    grid = make_grid(gen_true, nrow=8)
    img = grid.permute(1, 2, 0).cpu().numpy()
    if img.shape[2] == 1:
        img = img[:, :, 0]
    return img

def generate_samples_matplotlib(
    model,
    process,
    integrator,
    normalizer,
    config,
    dataset,
    device,
    n_samples=9,
    n_steps=None,
    cmap="gray"
):
    """
    Generate samples from the model and return as wandb.Image using matplotlib.
    
    This uses the same visualization approach as backward_final_samples from eval_plotting,
    displaying samples in a horizontal strip with matplotlib instead of torchvision grid.
    
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
        cmap: Colormap for grayscale images (default: "gray")
        
    Returns:
        wandb.Image object or None if failed
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
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
        stationary_sampler = StationarySampler(config, device=device)
        x0 = stationary_sampler((n_samples, dataset.C, dataset.H, dataset.W))
        
        # Run reverse process
        with torch.inference_mode():
            t_grid, X = solver.simulate(x0, n_steps=n_steps, seed=None)
        
        # Get final samples (latest time = reconstructed data)
        final_samples = X[-1]  # [n_samples, C, H, W]
        
        # Convert to list of numpy arrays (one per sample)
        final_samples_np = final_samples.cpu().numpy()
        samples_list = [final_samples_np[i] for i in range(n_samples)]
        
        # Create matplotlib figure - horizontal strip
        fig, axs = plt.subplots(1, n_samples, figsize=(n_samples * 2.5, 2.5), squeeze=False)
        axs = axs[0]  # 1D array of axes
        
        # Get visualization range from normalizer
        vmin, vmax = normalizer.get_visualization_range()
        
        for i in range(n_samples):
            ax = axs[i]
            
            # Get sample and denormalize
            sample = samples_list[i]  # [C, H, W]
            sample_torch = torch.from_numpy(sample)
            sample_denorm = normalizer.denormalize(sample_torch).cpu().numpy()
            
            # Clip to visualization range
            sample_denorm = np.clip(sample_denorm, vmin, vmax)
            
            # Handle channel dimension
            if sample_denorm.shape[0] == 1:
                # Grayscale: [1, H, W] -> [H, W]
                img = sample_denorm[0]
                ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            elif sample_denorm.shape[0] == 3:
                # RGB: [3, H, W] -> [H, W, 3]
                img = np.moveaxis(sample_denorm, 0, -1)
                img = np.clip(img, 0.0, 1.0)
                ax.imshow(img)
            else:
                raise ValueError(f"Unexpected channel count: {sample_denorm.shape[0]}")
            
            # Remove ticks and frame
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            
            # Add sample index
            ax.text(
                0.02, 0.02, f"{i}",
                transform=ax.transAxes, 
                ha="left", va="bottom",
                fontsize=9, 
                color="white",
                bbox=dict(facecolor="black", alpha=0.35, lw=0, pad=1.0)
            )
        
        # Tight layout with no padding
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
        
        # Convert figure to image for wandb
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_array = img_array[:, :, :3]  # Remove alpha channel
        
        plt.close(fig)
        
        return wandb.Image(img_array)
        
    except Exception as e:
        print(f"[red]Sample generation (matplotlib) failed: {e}[/red]", flush=True)
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        model.train()

def log_wandb_metrics(metrics_evo: dict):
    temp_dict = {}
    for key in metrics_evo.keys():
        if metrics_evo[key]:
            temp_dict[key] = metrics_evo[key][-1]
    wandb.log(temp_dict)

def create_header(metrics_evo: dict, width = None, external = None) -> str:
    header = "  " + "".join(f"{k:<{width}}" for k in metrics_evo.keys())
    if external:
        header += "".join(f"{k:<{width}}" for k in external)
    return f"[bold]{header}[/bold]"

def log_wandb_best(best_models: dict, config):
    wandb.summary["T"] = float(config.corruption.process_params.T)
    wandb.summary["power"] = float(config.corruption.process_params.alpha)
    for key, info in best_models.items():
        wandb.summary[f"best_{key}"] = float(info["value"])
        wandb.summary[f"best_{key}_epoch"] = int(info["epoch"])


def build_line(metrics_evo: dict, width = None, external_dict = None, METRIC_KEYS = METRIC_KEYS) -> str:
    cells = []
    for k in metrics_evo.keys():
        v_list = metrics_evo[k]
        v = v_list[-1] if v_list else "-"

        if isinstance(v, (float, np.floating)):
            cells.append(f"{v:<{width}.7f}")
        elif isinstance(v, int):
            cells.append(f"{v:<{width}d}")
        else:
            cells.append(f"{str(v):<{width}}")
    
    # Add external values
    if external_dict:
        for k, v in external_dict.items():
            if v is None:
                cells.append(f"{'-':<{width}}")
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:<{width}.7f}")
            elif isinstance(v, int):
                cells.append(f"{v:<{width}d}")
            else:
                cells.append(f"{str(v):<{width}}")

    return "  " + "".join(cells)


def collect_n_images(dataloader, n_images: int, device=None) -> torch.Tensor:
    """
    Collect exactly n_images from dataloader.
    Assumes each batch is a torch.Tensor of shape (B, C, H, W).
    Returns a torch.Tensor of shape (n_images, C, H, W).
    """
    xs = []
    collected = 0

    for batch in dataloader:
        x = batch
        if device is not None:
            x = x.to(device)

        need = n_images - collected
        if x.shape[0] > need:
            x = x[:need]

        xs.append(x)
        collected += x.shape[0]
        if collected >= n_images:
            break

    if collected != n_images:
        raise ValueError(f"Requested {n_images} images, but got {collected}. "
                         f"Check dataloader size / drop_last.")
    return torch.cat(xs, dim=0)


def init_wandb(config, petname, rundir):

    tags_from_config = getattr(config.wandb, 'tags', [])
    if isinstance(tags_from_config, str):
        tags_from_config = [tags_from_config]

    integrator_tag = config.corruption.integrator_cls
    if hasattr(config.corruption.integrator_params, 'r'):
        integrator_tag = f"{integrator_tag}_r{config.corruption.integrator_params.r}"

    tags = [config.corruption.process_cls, config.model.cls, integrator_tag] + tags_from_config

    wandb.init(
            project=getattr(config.wandb, 'project', 'diffusion-training'),
            entity=getattr(config.wandb, 'entity', None),
            name=petname,
            config=config.to_dict(),
            dir=rundir,
            mode=getattr(config.wandb, 'mode', 'online'),
            notes=getattr(config.wandb, 'notes', ''),
            tags=tags,
            group=str(config.corruption.process_params.alpha)
        )
    
    wandb.define_metric("epoch")
    wandb.define_metric("epoch_loss", step_metric="epoch")
    wandb.define_metric("generated_samples", step_metric="epoch")
    wandb.define_metric("best_loss", step_metric="epoch")
    
def maybe_update_best(key, value, epoch, best_models_dict, model):
    new_best = False
    if value < best_models_dict[key]["value"]:
        best_models_dict[key]["value"] = value
        best_models_dict[key]["epoch"] = epoch
        best_models_dict[key]["state_dict"] = copy.deepcopy(model.state_dict())
        new_best = True
    return new_best

def wandb_log_best_and_plot(model, proc, config, integrator, stationary_sampler, normalizer, best_models, C, H, W, epoch):
    import matplotlib.pyplot as plt

    was_training = model.training
    model.eval()
    orig_sd = copy.deepcopy(model.state_dict())

    keys = list(best_models.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 3 * len(keys)))
    if len(keys) == 1:
        axes = [axes]

    device = str(next(model.parameters()).device)

    for ax, k in zip(axes, keys):
        info = best_models[k]
        if info["state_dict"] is None:
            ax.set_title(f"{k} — no best recorded")
            ax.axis("off")
            continue
        model.load_state_dict(info["state_dict"])

        img = generate_8x8_grid(model, proc, integrator, normalizer, config, C, H, W, device)

        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(f"{k} @ epoch {info['epoch']}")
        ax.axis("off")

    model.load_state_dict(orig_sd)
    if was_training:
        model.train()

    wandb.log({"epoch": epoch, "generated_samples_best": wandb.Image(fig)})
    plt.close(fig)

def fill_metrics_results(metrics_evo, metrics_results):
    for key, value in metrics_results.items():
        metrics_evo[key].append(value)
    has_bad = any(not np.isfinite(v) for v in metrics_results.values())
    metrics_evo["nan"].append("NaN" if has_bad else "")

# def save_metrics_evo(metrics_evo: dict, run_dir: str, filename: str = "metrics_evo.json"):

#     filepath = os.path.join(run_dir, filename)
    
#     # Convert numpy types to native Python types
#     metrics_clean = {}
#     for key, values in metrics_evo.items():
#         metrics_clean[key] = [
#             float(v) if isinstance(v, (np.floating, np.integer)) 
#             else int(v) if isinstance(v, (np.int_, np.intc, np.intp))
#             else v
#             for v in values
#         ]
    
#     with open(filepath, "w") as f:
#         json.dump(metrics_clean, f, indent=2)
    
#     return filepath

def dump_dict_to_json(data_dict: dict, dir: str, filename: str = "data_dict.json"):
    filepath = os.path.join(dir, filename)
    
    data_clean = {}
    for key, value in data_dict.items():
        if isinstance(value, (np.floating, np.integer)):
            data_clean[key] = float(value) if isinstance(value, np.floating) else int(value)
        else:
            data_clean[key] = value
    
    with open(filepath, "w") as f:
        json.dump(data_clean, f, indent=2)
    
    return filepath
