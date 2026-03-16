"""
ml/metrics.py
-------------
W1 evaluator — exact Wasserstein-1 distance between two batches of images.

Usage:
    evaluator = W1Evaluator(real_true)   # once, before training loop
    w1_val = evaluator.compute(gen_true) # each eval
"""

import numpy as np
import torch
import ot


def _flatten(x: torch.Tensor) -> np.ndarray:
    """Flatten (N, C, H, W) tensor to (N, D) float64 numpy array."""
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return x.reshape(x.shape[0], -1).cpu().numpy().astype(np.float64)


class W1Evaluator:
    """
    Stores a fixed real batch and computes exact W1 against generated batches.
    Instantiate once before the training loop; call compute() each eval.
    """

    def __init__(self, real_true: torch.Tensor):
        self.real = _flatten(real_true)
        n = self.real.shape[0]
        self.w = np.ones(n) / n

    def compute(self, gen_true: torch.Tensor) -> float:
        gen = _flatten(gen_true)
        w_gen = np.ones(gen.shape[0]) / gen.shape[0]
        M = ot.dist(self.real, gen, metric="euclidean")
        return float(ot.emd2(self.w, w_gen, M, numItermax=1_000_000))