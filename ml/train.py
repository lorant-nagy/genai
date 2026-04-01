#train.py
import petname
import wandb
from rich import print
import numpy as np

import argparse
import os
import shutil
import sys
sys.path.append(os.path.abspath(".."))

import torch
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.float32)

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
    generate_samples_matplotlib,
    load_cfg
)

from utils.registry import REGISTRY

from ml.metrics import W1Evaluator

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

SIMPLE_LOG = {
    "T": config.corruption.process_params.T,
    "stepc": config.corruption.corruptor_params.n_steps,
    "sche": config.corruption.integrator_cls,
    "nx0": config.corruption.process_params.score_table_params.N_x0,
    "n_x": config.corruption.process_params.score_table_params.N_x,
    "n_t": config.corruption.process_params.score_table_params.N_t,
    "pow" : config.corruption.process_params.alpha,
}
for key, value in SIMPLE_LOG.items():
    petname_str += f"_{key}{value}"


adaptive_step = getattr(config.env, 'adaptive_step', False)
if adaptive_step:
    base = config.env.adaptive_base
    config.corruption.corruptor_params.n_steps = int(base * config.corruption.process_params.T)
else:
    print(f"[bold yellow]Using fixed number of steps:[/] [yellow]{config.corruption.corruptor_params.n_steps}[/]")

project_dir = os.path.join(config.env.results_dir, config.wandb.project)
run_dir = os.path.join(config.env.results_dir, config.wandb.project, "runs", petname_str + "_" + time_str)
table_dir = os.path.join(config.env.results_dir, "score_tables")
os.makedirs(run_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)
os.makedirs(os.path.join(run_dir, "states"), exist_ok=True)

# # # # # # # # B L O C K 2 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

try:

    # dump config
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        import yaml
        yaml.dump(config.to_dict(), f)

    init_wandb(config, petname_str, run_dir)

    wandb.log({"simple_log": SIMPLE_LOG})

    normalizer_name = config.dataset_train.normalizer_cls
    normalizer_parameters = config.dataset_train.normalizer_params
    normalizer = REGISTRY[normalizer_name](**normalizer_parameters.to_dict())

    dataset_name = config.dataset_train.dataset_cls
    dataset_parameters = config.dataset_train.dataset_params
    dataset_parameters.normalizer = normalizer
    dataset = REGISTRY[dataset_name](**dataset_parameters.to_dict())
    dataloader = DataLoader(dataset, **config.dataloader.to_dict())

    eval_dl_kwargs = dict(config.dataloader.to_dict())
    eval_dl_kwargs["shuffle"] = False      # MUST: same real subset across runs
    eval_dl_kwargs["drop_last"] = False
    eval_dl_kwargs["num_workers"] = 0      # strongly recommended for strict determinism
    eval_dataloader = DataLoader(dataset, **eval_dl_kwargs)

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
    stationary_sampler = StationarySampler(config, device=DEVICE)

    EVAL_SEED = int(getattr(config.eval, "seed", 0))  # if not in config, defaults to 0

    x0_eval = stationary_sampler(
        (config.eval.n_metric_samples, dataset.C, dataset.H, dataset.W),
        seed=EVAL_SEED
    )

    BENCHMARK_METRICS = ["w1"]

    print("[bold green]---------------------------------------------------------------[/bold green]")
    print(f"[bold green]RUN NAME : ------------------[/][yellow] {petname_str} [/][bold green]------------------[/bold green]")
    print("[bold green]---------------------------------------------------------------[/bold green]")

    print(f"\n[bold cyan]TRAINING:[/bold cyan]")

    average_loss_per_sample = 0.0
    loss_evo = []

    real_norm = collect_n_images(eval_dataloader, config.eval.n_metric_samples, device=DEVICE)
    real_true = normalizer.denormalize(real_norm).clamp(0, 1)
    w1_evaluator = W1Evaluator(real_true)

    best_models = {key : {'value': float('inf'), 'state_dict': None, 'epoch': 0} for key in BENCHMARK_METRICS}
    metrics_evo = {key: [] for key in ['epochs', 'loss', 'nan', 'w1']}

    print_tab_w = 16
    header = create_header(metrics_evo, print_tab_w, external = ["corrupt_img%", "corrupt_px%", "first_corrupt_step"])
    print(header)

    # # # # # # # # B L O C K 3 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    backward_cntr = 0
    loss_sum_interval = 0.0
    log_backward_freq = max(1, len(dataloader) // config.eval.evals_per_epoch)
    print(f"[bold blue]Evals per epoch:[/] [yellow]{config.eval.evals_per_epoch}[/] → log_backward_freq: {log_backward_freq}")
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    for epoch in range(config.train.n_epochs):

        for c, batch in enumerate(dataloader):
            backward_cntr += 1
            batch = batch.to(DEVICE)
            corrupted = corruptor(batch)
            score = proc.score(corrupted['x'], batch, corrupted['t'])
            prediction = model(corrupted['x'], corrupted['t'])
            output = loss_fn(prediction, score)

            loss_sum_interval += output.item()

            output.backward()
            optimizer.step()
            optimizer.zero_grad()

            if backward_cntr % log_backward_freq != 0:
                continue

            # # # # # # # # E V A L  B L O C K # # # # # # # # # # # # # # #

            avg_loss = loss_sum_interval / log_backward_freq
            loss_sum_interval = 0.0
            loss_evo.append(avg_loss)

            metrics_evo['epochs'].append(backward_cntr)
            metrics_evo['loss'].append(avg_loss)

            model.eval()
            CHW = (dataset.C, dataset.H, dataset.W)
            reverse_sde = make_reverse(proc, model, config.corruption.process_params.T, **config.reverse_params.to_dict(), CHW=CHW)
            backward_solver = SDESolver(reverse_sde, integrator)

            # ── Chunked reverse simulation ─────────────────────────────────
            # Run the reverse SDE in chunks to avoid OOM in the attention
            # layer (which is quadratic in batch size).  NaN/Inf diagnostics
            # are accumulated across chunks so the logged percentages are
            # identical to what a single-batch run would produce.
            EVAL_CHUNK = 256

            gen_norm_chunks        = []
            corrupt_img_flags      = []   # bool per image: has at least one bad pixel
            corrupt_px_sums        = []   # bad-pixel fraction per corrupted image
            first_appearance_list  = []   # first-bad-step per pixel, flattened

            n_steps_eval = config.corruption.corruptor_params.n_steps

            with torch.no_grad():
                for chunk in x0_eval.split(EVAL_CHUNK):
                    t_grid, X_chunk = backward_solver.simulate(
                        chunk,
                        n_steps=n_steps_eval,
                        seed=EVAL_SEED,
                    )
                    total_steps = len(X_chunk)
                    final        = X_chunk[-1].detach()          # (chunk, C, H, W)

                    # ── NaN first-appearance tracking ──────────────────────
                    first_app = torch.full_like(final, total_steps, dtype=torch.float32)
                    for step_idx, X_t in enumerate(X_chunk):
                        bad       = torch.isnan(X_t) | torch.isinf(X_t)
                        newly_bad = bad & (first_app == total_steps)
                        first_app[newly_bad] = step_idx

                    affected = first_app < total_steps
                    if affected.any():
                        first_appearance_list.append(first_app[affected])

                    # ── Per-image / per-pixel corrupt stats ────────────────
                    nan_mask   = torch.isnan(final) | torch.isinf(final)
                    per_img    = nan_mask.view(final.shape[0], -1)   # (chunk, C*H*W)
                    img_flags  = per_img.any(dim=1)                  # (chunk,) bool
                    corrupt_img_flags.append(img_flags)

                    bad_imgs = per_img[img_flags]
                    if bad_imgs.shape[0] > 0:
                        corrupt_px_sums.append(bad_imgs.float().mean(dim=1))  # per corrupted image

                    gen_norm_chunks.append(final)

            # ── Reassemble ────────────────────────────────────────────────
            gen_norm  = torch.cat(gen_norm_chunks, dim=0)           # (N, C, H, W)
            gen_true  = normalizer.denormalize(gen_norm).clamp(0, 1)

            # ── Aggregate NaN stats ───────────────────────────────────────
            all_flags       = torch.cat(corrupt_img_flags)          # (N,) bool
            corrupt_img_pct = all_flags.float().mean().item() * 100

            if corrupt_px_sums:
                corrupt_px_pct = torch.cat(corrupt_px_sums).mean().item() * 100
            else:
                corrupt_px_pct = 0.0

            if first_appearance_list:
                avg_first_appearance = torch.cat(first_appearance_list).mean().item()
            else:
                avg_first_appearance = total_steps

            metrics_results = {"w1": w1_evaluator.compute(gen_true)}
            fill_metrics_results(metrics_evo, metrics_results)

            sample_grid_matplotlib = generate_samples_matplotlib(
                model=model,
                process=proc,
                integrator=integrator,
                normalizer=normalizer,
                config=config,
                dataset=dataset,
                device=DEVICE,
                n_samples=8,
                n_steps=n_steps_eval,
            )

            model.train()
            # # e n d  o f  e v a l  b l o c k # # # # # # #

            if sample_grid_matplotlib is not None:
                wandb.log({"generated_samples": sample_grid_matplotlib, "backward_step": backward_cntr})

            for metric in BENCHMARK_METRICS:
                if metric in metrics_evo and len(metrics_evo[metric]) > 0:
                    maybe_update_best(metric, metrics_evo[metric][-1], backward_cntr, best_models, model)

            log_wandb_metrics(metrics_evo)

            external_dict = {
                "corrupt_img%": corrupt_img_pct,
                "corrupt_px%":  corrupt_px_pct,
                "first_corrupt_step": avg_first_appearance,
            }
            wandb.log(external_dict)

            line = build_line(metrics_evo, print_tab_w, external_dict=external_dict)
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
    #dump the simple log txt
    with open(os.path.join(run_dir, "simple_log.txt"), "a") as f:
        f.write(f"{petname_str}_{time_str} : {SIMPLE_LOG}\n")

    wandb.finish()
    torch.cuda.empty_cache()
    del model

    print("[bold green]---------------------------------------------------------------[/bold green]")
    print(f"[bold green]RUN ENDED -- NAME : ------------------[/][yellow] {petname_str} [/][bold green]------------------[/bold green]")
    print("[bold green]---------------------------------------------------------------[/bold green]")
except Exception:
    print("[bold red]Training failed — cleaning up run dir[/bold red]")
    try:
        wandb.finish(exit_code=1)
    except Exception:
        pass
    shutil.rmtree(run_dir, ignore_errors=True)
    raise