from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import torch

from diff.sim_core import SDESolver
from diff.integrator import EulerMaruyama  # or pass a different integrator
from diff.sim_core import Integrator  # for type hinting

@dataclass
class CorruptorConfig:
    n_steps: int
    mode: Literal["trajectory", "snapshot"] = "trajectory"
    per_sample_time: bool = True  # for 'snapshot' mode: draw a separate t for each sample
    device: Optional[torch.device] = None  # if None, will use process.device
    dtype: Optional[torch.dtype] = None    # if None, keep input dtype
    integrator: Optional[Integrator] = None  # if None, will use EulerMaruyama
    process: Optional[Any] = None  # placeholder for process type hinting

class Corruptor:
    """
    Input:
        - process: ItoProcess with .dim, .t0, .T, .device, .drift(x,t), .diffusion(x,t)
        - images:  torch.Tensor of shape (B, C, H, W).
        - seed

        * time first is assumed thorughout the code, (T, B, C, H, W)
    Output:
        If mode='trajectory':
            dict(
                x = (T_steps+1, B, C, H, W)
                t = (T_steps+1,)  # time grid in process time
            )

        If mode='snapshot':
            dict(
                x = (B, C, H, W),  # samples at chosen time(s)
                t = (B,) if per_sample_time else scalar tensor of shape ()
            )

    Notes:
        - No analytic marginals are used; everything is produced by Euler–Maruyama.
        - We simulate the full grid even in 'snapshot' mode : suboptimal, but simple.
    """

    def __init__(self, n_steps: int, mode: Literal["trajectory", "snapshot"] = "trajectory", per_sample_time: bool = True, integrator: Optional[EulerMaruyama] = None, process: Optional[Any] = None, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, return_time_zero_state: bool = False):
        
        self.n_steps = n_steps
        self.mode = mode
        self.per_sample_time = per_sample_time
        self.integrator = integrator
        self.process = process
        self.device = device
        self.dtype = dtype
        self.return_time_zero_state = return_time_zero_state

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, *, seed: Optional[int] = None) -> Dict[str, Any]:
        if images.ndim != 4:
            raise ValueError(f"images must be (B,C,H,W); got {tuple(images.shape)}")

        B, C, H, W = images.shape
        D = C * H * W

        device = self.device
        dtype = self.dtype

        x0 = images.to(device=device).view(B, D)

        # process dimension must match flattened dimension
        if getattr(self.process, "dim", None) is None or int(self.process.dim) != D:
            raise ValueError(f"process.dim ({getattr(self.process,'dim',None)}) must equal C*H*W ({D}).")

        solver = SDESolver(self.process, self.integrator)

        # simulate full trajectory
        t_grid, X = solver.simulate(
            x0,
            n_steps=self.n_steps,
            seed=seed,
            return_trajectory=True,
        )
        # X: (T_steps+1, B, D)
        Tn = X.shape[0]

        #  (T, B, C, H, W) is assumed
        X_img = X.view(Tn, B, C, H, W)

        if self.mode == "trajectory":
            out_x = X_img
            return {"x": out_x, "t": t_grid}

        elif self.mode == "snapshot":
            # uniformly sampling of time on the grid
            if self.per_sample_time:
                if not self.return_time_zero_state:
                    idx = torch.randint(low=1, high=Tn, size=(B,), device=device)  # we exclude t=0
                x_snap = X_img[idx, torch.arange(B)]
                t_snap = t_grid[idx]  # (B,)
            else:
                k = int(torch.randint(low=0, high=Tn, size=(1,), device=device).item())
                x_snap = X_img[k]     # (B, C, H, W)
                t_snap = t_grid[k]    # scalar tensor

            return {"x": x_snap, "t": t_snap}
        
        else:
            raise NotImplementedError(f"mode={self.mode} is not implemented.")
