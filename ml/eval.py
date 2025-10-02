from utils.config import load_cfg, infer_generic
from utils.registry import REGISTRY
from torch.utils.data import DataLoader
import torch
import sys
import os
sys.path.append(os.path.abspath(".."))
import argparse

from diff.sim_core import SDESolver
from diff.reverse import make_reverse
from diff.integrator import EulerMaruyama

from utils.samplers import standard_normal_BCHW

from utils.eval_plotting import plot_corruption_and_samples

import numpy as np

import matplotlib.pyplot as plt

from dataset import RectanglesDataset
from model import ScoreNet
from diff.sde import VPOU
from diff.corruptor import Corruptor
from diff.sim_core import SDESolver, ItoProcess
from diff.score import vp_ou_score


parser = argparse.ArgumentParser(description="Evaluation script")
parser.add_argument("config", type=str, help="Path to the config file")
parser.add_argument("state_dict", type=str, help="Path to the state dict")
parser.add_argument("eval_path", type=str, help="Path to save evaluation results")
args = parser.parse_args()


config = load_cfg(args.config)

dataset_name = config.dataset_eval.dataset_cls
dataset_parameters = config.dataset_eval.dataset_params
infer_generic(config, dataset_parameters)
dataset = REGISTRY[dataset_name](**dataset_parameters.to_dict())

dataloader = DataLoader(dataset, **config.dataloader.to_dict())

model_name = config.model.cls
model_parameters = config.model.model_params
model_parameters.in_channels = dataset.C
model = REGISTRY[model_name](**model_parameters.to_dict())
state_dict = torch.load(args.state_dict)
model.load_state_dict(state_dict)
model.to(device=config.generic.device)
model.eval()

os.makedirs(args.eval_path, exist_ok=True)

batch_of_data = next(iter(dataloader))
batch_of_data = batch_of_data.to(device=config.generic.device)

process_name = config.corruption.process_cls
process_parameters = config.corruption.process_params
image_space_dim = dataset.C * dataset.H * dataset.W
process_parameters.dim = image_space_dim
infer_generic(config, process_parameters)
proc = REGISTRY[process_name](**process_parameters.to_dict())

integrator_name = config.corruption.integrator_cls
integrator_parameters = config.corruption.integrator_params
integrator = REGISTRY[integrator_name](**integrator_parameters.to_dict())


corruptor_parameters = config.corruption.corruptor_params
infer_generic(config, corruptor_parameters)
corruptor_parameters.integrator = integrator
corruptor_parameters.process = proc
# override mode : . \to "trajectory"
corruptor_parameters.mode = "trajectory"
corruptor = Corruptor(**corruptor_parameters.to_dict())

n_samples = config.eval.n_samples
reverse_sde = make_reverse(proc, model, config.corruption.process_params.T, **config.reverse_params.to_dict())
backward_solver = SDESolver(reverse_sde, EulerMaruyama())
x0 = standard_normal_BCHW(n_samples, dataset.C, dataset.H, dataset.W, device=config.generic.device)
n_steps = config.eval.n_steps
print(f"-------> simulating backward trajectories {n_samples} samples with {n_steps} steps", flush=True)
with torch.inference_mode():
    t_grid, X = backward_solver.simulate(
                x0,
                n_steps=n_steps,
                return_trajectory=True,
            )

print(f"-------> plotting", flush=True)
plot_corruption_and_samples(
    corruptor=corruptor,
    batch_of_data=batch_of_data,
    eval_path=args.eval_path,
    cmap="gray",
    max_trajectories=5,
    sample_grid_count=9,
    show_time_header=True,
    mark_t0_indices=True,
    label_data_sample_indices=True,
    t_grid=t_grid,
    X=X[:, :9],
    n_time_cols=10,
)
