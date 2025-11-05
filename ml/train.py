
import argparse
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(".."))

from utils.config import load_cfg, infer_generic
from utils.registry import REGISTRY

from torch.utils.data import DataLoader
import torch
from diff.corruptor import Corruptor
from diff.sim_core import SDESolver, ItoProcess
# from diff.score import vp_ou_score

from dataset import RectanglesDataset
from model import ScoreNet
from diff.sde import VPOU

parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("config", type=str, help="Path to the config file")
parser.add_argument("results_path", type=str, help="Path to save training results")
args = parser.parse_args()

config = load_cfg(args.config)

dataset_name = config.dataset_train.dataset_cls
dataset_parameters = config.dataset_train.dataset_params
infer_generic(config, dataset_parameters)


dataset = REGISTRY[dataset_name](**dataset_parameters.to_dict())

dataloader_train = DataLoader(dataset, **config.dataloader.to_dict())

model_name = config.model.cls
model_parameters = config.model.model_params
model_parameters.in_channels = dataset.C

model = REGISTRY[model_name](**model_parameters.to_dict())

optimizer_name = config.train.optimizer_cls
optimizer_parameters = config.train.optimizer_params
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_parameters.to_dict())

process_name = config.corruption.process_cls
process_parameters = config.corruption.process_params
image_space_dim = dataset.C * dataset.H * dataset.W
infer_generic(config, process_parameters)
proc = REGISTRY[process_name](**process_parameters.to_dict())

integrator_name = config.corruption.integrator_cls
integrator_parameters = config.corruption.integrator_params
integrator = REGISTRY[integrator_name](**integrator_parameters.to_dict())

corruptor_parameters = config.corruption.corruptor_params
infer_generic(config, corruptor_parameters)
corruptor_parameters.integrator = integrator
corruptor_parameters.process = proc
corruptor = Corruptor(**corruptor_parameters.to_dict())

loss_fn = getattr(torch.nn, config.loss.cls)(**config.loss.loss_params.to_dict())

# print info on training
print(f"-- dataset: {dataset_name} with parameters {dataset_parameters}")
print(f"-- model: {model_name} with parameters {model_parameters}")
print(f"-- corruption process: {process_name} with parameters {process_parameters}")
print(f"-- corruption integrator: {integrator_name} with parameters {integrator_parameters}")
print(f"-- corruption parameters: {corruptor_parameters}")
print(f"-- optimizer: {optimizer_name} with parameters {optimizer_parameters}")
print(f"-- loss: {config.loss.cls} with parameters {config.loss.loss_params}")
print(f"-- training for {config.train.n_epochs} epochs")

print(f"*** training started")
best_model = None
best_loss = float('inf')
average_loss_per_sample = 0.0
loss_evo = []
for epoch in range(config.train.n_epochs):
    loss = 0.0
    for c,batch in enumerate(dataloader_train):
        corrupted = corruptor(batch)
        score = proc.score(corrupted['x'], batch, corrupted['t'])
        prediction = model(corrupted['x'], corrupted['t'])
        output = loss_fn(prediction, score)
        loss += output
        output.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"batch {c+1}/{len(dataloader_train)}", end="\r", flush=True)
    average_loss_per_sample = loss.item()/len(dataloader_train)
    print(f"Epoch {epoch+1}/{config.train.n_epochs} || Loss: {average_loss_per_sample:.6f}", flush=True)
    loss_evo.append(average_loss_per_sample)
    if average_loss_per_sample < best_loss:
        best_loss = average_loss_per_sample
        best_model = model.state_dict()
print(f"*** training finished", flush=True)

# save model
torch.save(best_model, os.path.join(args.results_path, "best_model.pth"))

# report training
print("REPORT ON TRAINING:")
print(f"Best Loss: {best_loss:.6f}", flush=True)

# evaluate
print("*** starting evaluation", flush=True)
eval_script_path = os.path.join(os.path.dirname(__file__), "eval.py")
os.system(f"python {eval_script_path} {args.config} {os.path.join(args.results_path, 'best_model.pth')} {args.results_path}")
print("*** evaluation finished", flush=True)

#  plots
plt.figure()
plt.plot(loss_evo)
plt.xlabel("Epoch")
plt.ylabel("Average Loss per Sample")
plt.title("Loss Evolution During Training")
plt.grid()
plt.savefig(os.path.join(args.results_path, "loss_evolution.png"))
plt.close()