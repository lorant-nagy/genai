import torch
from abc import ABC, abstractmethod
from utils.registry import register

class Integrator(ABC):
    @abstractmethod
    def step(self, process, x: torch.Tensor, t: float, dt: float, dW: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

@register
class EulerMaruyama(Integrator):
    
    def __init__(self, **kwargs):
        """Accept any parameters for compatibility with config system."""
        pass
    
    def step(self, process, x: torch.Tensor, t: float, dt: float, dW: torch.Tensor) -> torch.Tensor:
        drift = process.drift(x, t)
        diffusion = process.diffusion(x, t)
        return x + drift * dt + diffusion * dW


@register
class TamedEulerTUSLA(Integrator):
    
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