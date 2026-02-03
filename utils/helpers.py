#helpers.py
import wandb
import copy
import torch
import numpy as np

def log_wandb_metrics(metrics_evo: dict):
    temp_dict = {}
    for key in metrics_evo.keys():
        if metrics_evo[key]:
            temp_dict[key] = metrics_evo[key][-1]
    wandb.log(temp_dict)

def create_header(metrics_evo: dict, width: int = 15) -> str:
    header = "  " + "".join(f"{k:<{width}}" for k in metrics_evo.keys())
    return f"[bold]{header}[/bold]"


def build_line(metrics_evo: dict, width: int = 15) -> str:
    cells = []
    for k in metrics_evo.keys():
        v_list = metrics_evo[k]
        v = v_list[-1] if v_list else ""

        if isinstance(v, (float, np.floating)):
            cells.append(f"{v:<{width}.4f}")
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
            tags=tags
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
