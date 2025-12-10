from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import torch

from diff.sim_core import SDESolver
from diff.integrator import EulerMaruyama
from diff.sim_core import Integrator

from utils.globals import DEVICE, DTYPE

@dataclass
class CorruptorConfig:
    n_steps: int
    mode: Literal["trajectory", "snapshot"] = "trajectory"
    per_sample_time: bool = True  # for 'snapshot' mode: draw a separate t for each sample
    integrator: Optional[Integrator] = None
    process: Optional[Any] = None  # placeholder for process type hinting

class Corruptor:
    """
    Input:
        - process: ItoProcess with .dim, .t0, .T, .device, .drift(x,t), .diffusion(x,t)
        - images:  torch.Tensor of shape (B, C, H, W)
        * time first : (T, B, C, H, W)
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
    """

    def __init__(self,
                n_steps: int,
                mode: Literal["trajectory", "snapshot"] = "trajectory",
                per_sample_time: bool = True,
                integrator: Optional[Integrator] = None,
                process: Optional[Any] = None,
                return_time_zero_state: bool = False
                 ) -> None:
        
        self.n_steps = n_steps
        self.mode = mode
        self.per_sample_time = per_sample_time
        self.integrator = integrator
        self.process = process
        self.device = DEVICE
        self.dtype = DTYPE
        self.return_time_zero_state = return_time_zero_state

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, *, seed: Optional[int] = None) -> Dict[str, Any]:

        B, C, H, W = images.shape

        device = self.device
        dtype = self.dtype

        x0 = images.to(device=device)
        solver = SDESolver(self.process, self.integrator)

        # simulate full trajectory
        t_grid, X = solver.simulate(
            x0,
            n_steps=self.n_steps,
            seed=seed
        )
        # X: [T_steps+1, B, C, H, W]
        Tn = X.shape[0]

        #  (T, B, C, H, W) is assumed
        X_img = X.view(Tn, B, C, H, W)

        if self.mode == "trajectory":
            out_x = X_img
            return {"x": out_x, "t": t_grid}

        elif self.mode == "snapshot":
            # uniformly sampling of time on the grid
            if self.per_sample_time:
                # X: (Tn, B, C, H, W), t_grid: (Tn,)
                idx = torch.randint(1, Tn, (B,), device=X.device) if not self.return_time_zero_state else \
                    torch.randint(0, Tn, (B,), device=X.device)
                x_snap = X[idx, torch.arange(B)]          # (B, C, H, W)
                t_snap = t_grid[idx]                      # (B,)
            else:
                k = int(torch.randint(0, Tn, (1,), device=X.device))
                x_snap = X[k]                              # (B, C, H, W)
                t_snap = t_grid[k]                         # scalar

            return {"x": x_snap, "t": t_snap}
        
        else:
            raise NotImplementedError(f"mode={self.mode} is not implemented.")
