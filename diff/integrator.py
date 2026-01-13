import torch
from abc import ABC, abstractmethod
from utils.registry import register


class Integrator(ABC):
    @abstractmethod
    def step(self, process, x: torch.Tensor, t: float, dt: float, dW: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@register
class EulerMaruyama(Integrator):
    def __init__(self):
        pass

    def step(self, process, x: torch.Tensor, t: float, dt: float, dW: torch.Tensor) -> torch.Tensor:
        drift = process.drift(x, t)
        diffusion = process.diffusion(x, t)
        return x + drift * dt + diffusion * dW


@register
class StateNormTamedEuler(Integrator):
    # x_{n+1} = x_n + dt * b(x_n,t_n)/(1 + sqrt(dt)*||x_n||^{2r}) + sigma(x_n,t_n) dW_n
    def __init__(self, r: float = 1.0, eps: float = 0.0):
        self.r = float(r)
        self.eps = float(eps)

    def step(self, process, x: torch.Tensor, t: float, dt: float, dW: torch.Tensor) -> torch.Tensor:
        drift = process.drift(x, t)
        diffusion = process.diffusion(x, t)

        B = x.shape[0]
        x_norm = torch.linalg.vector_norm(x.reshape(B, -1), dim=1, keepdim=True)  # (B,1)
        x_norm_2r = (x_norm + self.eps) ** (2.0 * self.r)                         # (B,1)
        x_norm_2r = x_norm_2r.view(B, 1, 1, 1)

        taming = 1.0 + (dt ** 0.5) * x_norm_2r
        return x + (drift / taming) * dt + diffusion * dW


@register
class DriftNormTamedEuler(Integrator):
    # x_{n+1} = x_n + dt * b(x_n,t_n)/(1 + dt*||b(x_n,t_n)||) + sigma(x_n,t_n) dW_n
    def __init__(self, eps: float = 0.0):
        self.eps = float(eps)

    def step(self, process, x: torch.Tensor, t: float, dt: float, dW: torch.Tensor) -> torch.Tensor:
        drift = process.drift(x, t)
        diffusion = process.diffusion(x, t)

        B = x.shape[0]
        drift_norm = torch.linalg.vector_norm(drift.reshape(B, -1), dim=1, keepdim=True)  # (B,1)
        drift_norm = drift_norm.view(B, 1, 1, 1)

        taming = 1.0 + dt * (drift_norm + self.eps)
        return x + (drift / taming) * dt + diffusion * dW
