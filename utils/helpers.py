#helpers.py
import wandb
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from diff.reverse import make_reverse
from diff.sim_core import SDESolver

def log_wandb_metrics(metrics_evo: dict):
    temp_dict = {}
    for key in metrics_evo.keys():
        if metrics_evo[key]:
            temp_dict[key] = metrics_evo[key][-1]
    wandb.log(temp_dict)

def create_header(metrics_evo: dict, width: int = 15) -> str:
    header = "  " + "".join(f"{k:<{width}}" for k in metrics_evo.keys())
    return f"[bold]{header}[/bold]"

def log_wandb_best(best_models: dict, config):
    wandb.summary["T"] = float(config.corruption.process_params.T)
    wandb.summary["power"] = float(config.corruption.process_params.power)
    for key, info in best_models.items():
        wandb.summary[f"best_{key}"] = float(info["value"])
        wandb.summary[f"best_{key}_epoch"] = int(info["epoch"])


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

def wandb_log_best_and_plot(
    model, proc, config, integrator, stationary_sampler, normalizer, best_models, C, H, W, epoch
):
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    from diff.reverse import make_reverse
    from diff.sim_core import SDESolver

    N = 64  # hardcoded count (MINIMAL)
    CHW = (C, H, W)

    was_training = model.training
    model.eval()
    orig_sd = copy.deepcopy(model.state_dict())

    reverse_sde = make_reverse(proc, model, config.corruption.process_params.T, **config.reverse_params.to_dict(), CHW=CHW)
    backward_solver = SDESolver(reverse_sde, integrator)

    keys = list(best_models.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 3 * len(keys)))
    if len(keys) == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, k in zip(axes, keys):
            info = best_models[k]
            model.load_state_dict(info["state_dict"])

            x0 = stationary_sampler((N, C, H, W))
            _, X = backward_solver.simulate(x0, n_steps=config.corruption.corruptor_params.n_steps)

            gen_norm = X[-1].detach()
            del X
            gen_true = normalizer.denormalize(gen_norm).clamp(0, 1)

            grid = make_grid(gen_true, nrow=8)  # (C,H,W)
            img = grid.permute(1, 2, 0).cpu().numpy()
            if img.shape[2] == 1:
                img = img[:, :, 0]
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