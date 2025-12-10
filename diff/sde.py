import torch
from typing import Union, Callable, Optional
from diff.sim_core import ItoProcess
from utils.registry import register
import os
from diff.score_table import ScoreTableConfig, build_score_table, ScoreTable

@register
class VPOU(ItoProcess):

    def __init__(
        self,
        t0: float = 0.0,
        T: float = 1.0,
        beta: float = 1.0,
        device: str = "cpu",
        dtype: Union[str, torch.dtype, None] = None,
    ) -> None:
        
        super().__init__(t0=t0, T=T, device=device)
        
        self.beta = torch.as_tensor(beta, device=self.device)
        self._sqrt_beta = self.beta ** 0.5

    def drift(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return -0.5 * self.beta * x

    def diffusion(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return self._sqrt_beta * torch.ones_like(x)
    
    # vpouscore
    def score(self, x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        VP-OU (constant beta) conditional score.
        Maps ((B,C,H,W), (B,C,H,W), t) -> (B,C,H,W).
        """
        assert x_t.shape == x0.shape and x_t.ndim == 4, "x_t and x0 must be (B,C,H,W)"
        B = x_t.shape[0]
        t = t.view(B, 1, 1, 1)
        a = torch.exp(-0.5 * self.beta * t)
        var = torch.clamp(1.0 - a * a, min=1e-12)
        return -(x_t - a * x0) / var
    
    def sample_stationary(self, shape: tuple, device: Optional[str] = None) -> torch.Tensor:
        if device is None:
            device = self.device
        return torch.randn(shape, device=device)


@register
class SuperlinearLangevin(ItoProcess):
    r"""
    Superlinear Langevin: dX_t = -(c_α |X_t|^α sign(X_t) + c_0 X_t) dt + σ dW_t
    
    Samples from potential U(x) = (c_α/(α+1))|x|^(α+1) + (c_0/2)x^2
    
    Automatically builds or loads conditional score table on initialization.
    Provides stationary distribution sampling for reverse process initialization.
    """
    
    def __init__(
        self,
        t0: float = 0.0,
        T: float = 1.0,
        alpha: float = 3.0,
        c_alpha: float = 1.0,
        c_0: float = 0.5,
        sigma: float = 1.0,
        device: str = "cpu",
        score_table_params: Optional[dict] = None,
        table_path: str = None,
    ) -> None:
        super().__init__(t0=t0, T=T, device=device)
        
        # SDE parameters
        self.alpha = alpha
        self.c_alpha = c_alpha
        self.c_0 = c_0
        self.sigma = sigma
        
        self.score_table_obj = None
        
        if score_table_params is not None:
            if table_path is None:
                raise ValueError("path must be provided when score_table_params is given.")
            
            # Ensure directory exists
            os.makedirs(table_path, exist_ok=True)
            table_file_path = os.path.join(table_path, "score_table.pt")
            
            # Build ScoreTableConfig from the nested dict
            score_config = ScoreTableConfig(
                **score_table_params,
                alpha=self.alpha,
                c_alpha=self.c_alpha,
                c_0=self.c_0,
                sigma=self.sigma,
                T=self.T,
            )
            
            # Load or build
            if os.path.exists(table_file_path):
                print(f"Loading score table from {table_file_path}...")
                self.score_table_obj = ScoreTable.load(table_file_path, self)
                print("Score table loaded successfully.")
            else:
                print(f"Score table not found at {table_file_path}.")
                print(f"Building score table (this may take a while)...")
                self.score_table_obj = build_score_table(
                    self, 
                    score_config, 
                    save_path=table_file_path
                )
                print(f"Score table built and saved to {table_file_path}.")
    
    def drift(self, x: torch.Tensor, t: float) -> torch.Tensor:
        power_term = torch.pow(torch.abs(x), self.alpha) * torch.sign(x)
        return -(self.c_alpha * power_term + self.c_0 * x)
    
    def diffusion(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return self.sigma * torch.ones_like(x)
    
    def score(self, x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Conditional score using precomputed table.
        
        Args:
            x_t: (B, C, H, W) noisy images
            x0: (B, C, H, W) clean images
            t: (B,) time values
            
        Returns:
            (B, C, H, W) conditional scores
        """
        if self.score_table_obj is None:
            raise RuntimeError(
                "Score table not initialized. Add 'score_table_params' and 'path' "
                "to process config in YAML."
            )
        
        B, C, H, W = x_t.shape
        
        # Flatten to (B, d)
        x_t_flat = x_t.reshape(B, -1)
        x0_flat = x0.reshape(B, -1)
        
        # Query table (vectorized)
        score_flat = self.score_table_obj.interpolator.query(
            x_t_flat, t, x0_flat
        )
        
        # Reshape back
        return score_flat.reshape(B, C, H, W)