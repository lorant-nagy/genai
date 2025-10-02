from __future__ import annotations
import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from diff.integrator import Integrator

# torch.set_default_dtype(torch.float32)

class ItoProcess(ABC):
    """Base Itô process in R^d with fixed horizon.

    Expects inputs x with shape [B, d] (flat vectors).
    If you have images or structured data, flatten to R^d before using the solver.
    """

    def __init__(
        self,
        dim: int,   # dimension d of the flat state space R^d
        t0: float = 0.0,
        T: float = 1.0,
        device: str = "cpu",
    ) -> None:
        assert T > t0, "Require T > t0."
        self.dim = int(dim)
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
    """Fixed-step SDE simulator in flat space R^d.

    Accepts initial conditions x0 with shape [B, d] only.
    For images/structured data, flatten to [B, d] and set process.dim = d.
    """

    def __init__(self, process: ItoProcess, integrator: Integrator) -> None:
        self.process = process
        self.integrator = integrator
        self.device = process.device

    def simulate(
        self,
        x0: torch.Tensor,   # Initial batch of flat states, shape [B, d]
        n_steps: int,
        seed: Optional[int] = None,
        return_trajectory: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate SDE trajectories starting from x0.

        Args:
            x0: Initial conditions with shape [B, d]  (enforced)
            n_steps: Number of time steps
            seed: Random seed for reproducibility
            return_trajectory: If True, return full trajectory; if False, only final state

        Returns:
            t_grid: Time grid of shape [n_steps + 1]
            X: Trajectories with shape [n_steps + 1, B, d]
               or just final state [B, d] if return_trajectory=False
        """
        # Move x0 to device and enforce flat [B, d] shape
        x0 = x0.to(self.device)
        assert x0.ndim == 2, f"x0 must be (B, d); got {tuple(x0.shape)}"
        B, d = x0.shape
        assert d == int(self.process.dim), (
            f"process.dim ({self.process.dim}) must equal d ({d})."
        )

        # Time discretization
        t0 = self.process.t0
        T  = self.process.T
        dt = (T - t0) / float(n_steps)
        sqrt_dt = dt ** 0.5
        t_grid = torch.linspace(t0, T, n_steps + 1, device=self.device)

        # Trajectory storage
        if return_trajectory:
            X = torch.empty(n_steps + 1, B, d, device=self.device, dtype=x0.dtype)
            X[0] = x0

        # RNG
        rng = torch.Generator(device=self.device)
        if seed is not None:
            rng.manual_seed(int(seed))

        # Simulate
        x = x0.clone()
        with torch.inference_mode():
            for k in range(n_steps):
                t = float(t_grid[k])
                dW = torch.randn(x.shape, device=self.device, generator=rng, dtype=x.dtype) * sqrt_dt
                x  = self.integrator.step(self.process, x, t, dt, dW)
                if return_trajectory:
                    X[k + 1] = x

        if return_trajectory:
            return t_grid, X
        else:
            return t_grid, x
