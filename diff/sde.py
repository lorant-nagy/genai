import torch
from typing import Union, Callable
from diff.sim_core import ItoProcess
from utils.registry import register
from typing import Optional

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
    

@register
class SuperlinearLangevin(ItoProcess):
    r"""
    Superlinear Langevin: dX_t = -(c_α |X_t|^α sign(X_t) + c_0 X_t) dt + σ dW_t
    
    Samples from potential U(x) = (c_α/(α+1))|x|^(α+1) + (c_0/2)x^2
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
    ) -> None:
        super().__init__(t0=t0, T=T, device=device)
        self.alpha = alpha
        self.c_alpha = c_alpha
        self.c_0 = c_0
        self.sigma = sigma
    
    def drift(self, x: torch.Tensor, t: float) -> torch.Tensor:
        power_term = torch.pow(torch.abs(x), self.alpha) * torch.sign(x)
        return -(self.c_alpha * power_term + self.c_0 * x)
    
    def diffusion(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return self.sigma * torch.ones_like(x)
    
    def score(self, x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, steps: int =1) -> torch.Tensor:
        pass

