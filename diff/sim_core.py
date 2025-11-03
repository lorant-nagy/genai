from __future__ import annotations
import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from diff.integrator import Integrator

# torch.set_default_dtype(torch.float32)

class ItoProcess(ABC):

    def __init__(
        self,
        t0: float = 0.0,
        T: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.t0 = float(t0)
        self.T  = float(T)
        self.device = device

    @abstractmethod
    def drift(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Return b(x, t) with same shape as x (i.e., [B, d])."""
        raise NotImplementedError

    @abstractmethod
    def diffusion(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Return σ(x, t) with same shape as x (or broadcastable to [B, d])."""
        raise NotImplementedError


class SDESolver:

    def __init__(self, process: ItoProcess, integrator: Integrator) -> None:
        self.process = process
        self.integrator = integrator
        self.device = process.device

    def simulate(
        self,
        x0: torch.Tensor,
        n_steps: int,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            t_grid: Time grid of shape [n_steps + 1]
            X: Trajectories with shape [n_steps + 1, B, C, H, W]
        """
        x0 = x0.to(self.device)
        t0 = self.process.t0
        T  = self.process.T
        dt = (T - t0) / float(n_steps)
        sqrt_dt = dt ** 0.5
        t_grid = torch.linspace(t0, T, n_steps + 1, device=self.device)

        X = torch.empty(n_steps + 1, x0.shape[0], *x0.shape[1:], device=self.device, dtype=x0.dtype)
        X[0] = x0

        # RNG
        rng = torch.Generator(device=self.device)
        if seed is not None:
            rng.manual_seed(int(seed))

        # Simulate
        x = x0.clone()
        with torch.no_grad():
            for k in range(n_steps):
                t = float(t_grid[k])
                dW = torch.randn(x.shape, device=self.device, generator=rng, dtype=x.dtype) * sqrt_dt
                x  = self.integrator.step(self.process, x, t, dt, dW)
                X[k + 1] = x

        return t_grid, X