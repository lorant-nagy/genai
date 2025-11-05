import torch
from abc import ABC, abstractmethod
from utils.registry import register

class Integrator(ABC):
    """Base class for integrators."""
    @abstractmethod
    def step(self, process, x: torch.Tensor, t: float, dt: float, dW: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

@register
class EulerMaruyama(Integrator):
    
    def step(self, process, x: torch.Tensor, t: float, dt: float, dW: torch.Tensor) -> torch.Tensor:
        """
        Step: x_{n+1} = x_n + b(x_n, t_n)dt + σ(x_n, t_n)dW_n

        Args:
            process: ItoProcess instance
            x: Current state with shape [batch_size, ...] 
            t: Current time
            dt: Time step size
            dW: Brownian increment with same shape as x
            
        Returns:
            Next state with same shape as x
        """
        drift = process.drift(x, t)
        diffusion = process.diffusion(x, t)
        return x + drift * dt + diffusion * dW


@register
class TamedEulerTUSLA(Integrator):
    r"""
    TUSLA: x_{n+1} = x_n + drift / (1 + √dt |x_n|^{2r}) * dt + diffusion * dW
    
    Prevents explosion for superlinear drift. Use r ≥ α/2 where drift ~ |x|^α.
    """
    
    def __init__(self, r: float = 1.0):
        self.r = r
    
    def step(self, process, x: torch.Tensor, t: float, dt: float, dW: torch.Tensor) -> torch.Tensor:
        drift = process.drift(x, t)
        diffusion = process.diffusion(x, t)
        
        # Taming: compute |x|^{2r} per sample, shape (B,1,1,1)
        B = x.shape[0]
        x_norm_2r = torch.linalg.vector_norm(x.reshape(B, -1), dim=1, keepdim=True) ** (2 * self.r)
        x_norm_2r = x_norm_2r.view(B, 1, 1, 1)
        
        taming = 1.0 + (dt ** 0.5) * x_norm_2r
        
        return x + (drift / taming) * dt + diffusion * dW