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
from copy import deepcopy
import time
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
    filter_corrupt_images,
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

    # ── EMA shadow model ──────────────────────────────────────────────────────
    EMA_DECAY = 0.999
    ema_model = deepcopy(model)
    ema_model.eval()   # always in eval mode, never receives gradients

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
    metrics_evo = {key: [] for key in ['epochs', 'loss', 'nan', 'w1', 'trash%']}

    print_tab_w = 16
    header = create_header(metrics_evo, print_tab_w, external = ["corrupt_img%", "corrupt_px%", "first_corrupt_step"])
    print(header)

    # # # # # # # # B L O C K 3 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    backward_cntr = 0
    loss_sum_interval = 0.0
    log_backward_freq = max(1, len(dataloader) // config.eval.evals_per_epoch)
    eval_from = getattr(config.eval, "eval_from", 1)
    epoch_true = 0
    print(f"[bold blue]Evals per epoch:[/] [yellow]{config.eval.evals_per_epoch}[/] → log_backward_freq: {log_backward_freq}")
    print(f"[bold blue]Eval from epoch:[/] [yellow]{eval_from}[/]")
    t_train_start = time.time()
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    for epoch in range(config.train.n_epochs):
        epoch_true += 1
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

            # ── EMA update ────────────────────────────────────────────────
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.mul_(EMA_DECAY).add_(p, alpha=1.0 - EMA_DECAY)

            if backward_cntr % log_backward_freq != 0:
                continue

            # # # # # # # # E V A L  B L O C K # # # # # # # # # # # # # # #
            t_eval_start = time.time()

            avg_loss = loss_sum_interval / log_backward_freq
            loss_sum_interval = 0.0
            loss_evo.append(avg_loss)

            metrics_evo['epochs'].append(backward_cntr)
            metrics_evo['loss'].append(avg_loss)

            if epoch_true < eval_from:
                # warmup: skip expensive eval, keep list lengths in sync
                metrics_evo['w1'].append(None)
                metrics_evo['nan'].append("")
                metrics_evo['trash%'].append(None)
                external_dict = {
                    "corrupt_img%":      None,
                    "corrupt_px%":       None,
                    "first_corrupt_step": None,
                }
                log_wandb_metrics(metrics_evo)
                line = build_line(metrics_evo, print_tab_w, external_dict=external_dict)
                print(line)
                t_train_start = time.time()
                continue

            model.eval()
            CHW = (dataset.C, dataset.H, dataset.W)
            reverse_sde = make_reverse(proc, ema_model, config.corruption.process_params.T, **config.reverse_params.to_dict(), CHW=CHW)
            backward_solver = SDESolver(reverse_sde, integrator)

            # ── Chunked + multi-seed reverse simulation ───────────────────
            # Each seed draws a fresh x0 from the stationary distribution and
            # runs the full chunked reverse SDE.  W1 is computed on the pooled
            # generated samples (n_metric_samples × W1_N_SEEDS), reducing
            # variance.  NaN/corrupt stats are averaged across seeds.
            EVAL_CHUNK = 256
            W1_N_SEEDS = 1

            n_steps_eval = config.corruption.corruptor_params.n_steps

            all_gen_true         = []
            per_seed_corrupt_img = []
            per_seed_corrupt_px  = []
            per_seed_first_step  = []
            per_seed_trash_pct   = []

            with torch.no_grad():
                for seed_offset in range(W1_N_SEEDS):
                    seed = EVAL_SEED + seed_offset

                    # sample x0 in chunks to avoid OOM in the forward SDE
                    x0_chunks = []
                    for i in range(0, config.eval.n_metric_samples, EVAL_CHUNK):
                        chunk_size = min(EVAL_CHUNK, config.eval.n_metric_samples - i)
                        x0_chunks.append(stationary_sampler(
                            (chunk_size, dataset.C, dataset.H, dataset.W),
                            seed=seed + i,   # offset seed per chunk for diversity
                        ))
                    x0_seed = torch.cat(x0_chunks, dim=0)

                    gen_norm_chunks       = []
                    corrupt_img_flags     = []
                    corrupt_px_sums       = []
                    first_appearance_list = []

                    for chunk in x0_seed.split(EVAL_CHUNK):
                        t_grid, X_chunk = backward_solver.simulate(
                            chunk,
                            n_steps=n_steps_eval,
                            seed=seed,
                        )
                        total_steps = len(X_chunk)
                        final       = X_chunk[-1].detach()

                        # ── NaN first-appearance tracking ──────────────
                        first_app = torch.full_like(final, total_steps, dtype=torch.float32)
                        for step_idx, X_t in enumerate(X_chunk):
                            bad       = torch.isnan(X_t) | torch.isinf(X_t)
                            newly_bad = bad & (first_app == total_steps)
                            first_app[newly_bad] = step_idx

                        affected = first_app < total_steps
                        if affected.any():
                            first_appearance_list.append(first_app[affected])

                        nan_mask  = torch.isnan(final) | torch.isinf(final)
                        per_img   = nan_mask.view(final.shape[0], -1)
                        img_flags = per_img.any(dim=1)
                        corrupt_img_flags.append(img_flags)

                        bad_imgs = per_img[img_flags]
                        if bad_imgs.shape[0] > 0:
                            corrupt_px_sums.append(bad_imgs.float().mean(dim=1))

                        gen_norm_chunks.append(final)

                    # ── Reassemble this seed ────────────────────────────
                    gen_norm = torch.cat(gen_norm_chunks, dim=0)
                    gen_true = normalizer.denormalize(gen_norm).clamp(0, 1)
                    gen_true_clean, trash_pct = filter_corrupt_images(gen_true, threshold=0.10)
                    all_gen_true.append(gen_true_clean)
                    per_seed_trash_pct.append(trash_pct)

                    all_flags = torch.cat(corrupt_img_flags)
                    per_seed_corrupt_img.append(all_flags.float().mean().item() * 100)

                    if corrupt_px_sums:
                        per_seed_corrupt_px.append(torch.cat(corrupt_px_sums).mean().item() * 100)
                    else:
                        per_seed_corrupt_px.append(0.0)

                    if first_appearance_list:
                        per_seed_first_step.append(torch.cat(first_appearance_list).mean().item())
                    else:
                        per_seed_first_step.append(float(total_steps))

            # ── Pool across seeds and compute metrics ─────────────────────
            gen_true_pooled      = torch.cat(all_gen_true, dim=0)
            corrupt_img_pct      = float(np.mean(per_seed_corrupt_img))
            corrupt_px_pct       = float(np.mean(per_seed_corrupt_px))
            avg_first_appearance = float(np.mean(per_seed_first_step))
            metrics_evo['trash%'].append(float(np.mean(per_seed_trash_pct)))

            external_dict = {
                "corrupt_img%":       corrupt_img_pct,
                "corrupt_px%":        corrupt_px_pct,
                "first_corrupt_step": avg_first_appearance,
            }
            wandb.log(external_dict)

            if gen_true_pooled.shape[0] == 0:
                print("[bold red]All generated images were filtered as corrupt — skipping W1.[/bold red]")
                metrics_evo['w1'].append(None)
                metrics_evo['nan'].append("")
            else:
                metrics_results = {"w1": w1_evaluator.compute(gen_true_pooled)}
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
            torch.cuda.empty_cache()

            if sample_grid_matplotlib is not None:
                wandb.log({"generated_samples": sample_grid_matplotlib, "backward_step": backward_cntr})

            for metric in BENCHMARK_METRICS:
                if metric in metrics_evo and len(metrics_evo[metric]) > 0:
                    val = metrics_evo[metric][-1]
                    if val is not None:
                        maybe_update_best(metric, val, backward_cntr, best_models, ema_model)

            log_wandb_metrics(metrics_evo)

            t_eval_end   = time.time()
            t_eval_s     = t_eval_end - t_eval_start
            t_train_s    = t_eval_start - t_train_start
            timing_str   = f"  [train {t_train_s:.0f}s | eval {t_eval_s:.0f}s]"
            t_train_start = t_eval_end

            line = build_line(metrics_evo, print_tab_w, external_dict=external_dict)
            print(line + timing_str)

    # # # # # # # # B L O C K 4 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    log_wandb_best(best_models, config)

    wandb_log_best_and_plot(
        model=ema_model,
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