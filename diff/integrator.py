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


# class TamedEuler(Integrator):
#     """
#     Tamed Euler integrator (Sabanis, 2013) for SDEs with superlinear drift.
    
#     The taming prevents numerical explosion by damping large drift values:
#         x_{n+1} = x_n + b_tamed(x_n, t_n)dt + σ(x_n, t_n)dW_n
#     where
#         b_tamed = b / (1 + dt^α ||b||)
    
#     with 0 < α ≤ 1/2 controlling the taming strength.
#     """
    
#     def __init__(self, alpha: float = 0.5, eps: float = 1e-12):
#         """
#         Args:
#             alpha: Taming exponent (0 < α ≤ 0.5). Smaller values = stronger taming.
#             eps: Small constant for numerical stability
#         """
#         if not (0.0 < float(alpha) <= 0.5):
#             raise ValueError("alpha must be in (0, 0.5]. See Sabanis (2013) for theory.")
#         self.alpha = float(alpha)
#         self.eps = float(eps)

#     def step(self, process, x: torch.Tensor, t: float, dt: float, dW: torch.Tensor) -> torch.Tensor:
#         """
#         Single tamed integration step.
        
#         Args:
#             process: ItoProcess instance
#             x: Current state with shape [batch_size, ...] 
#             t: Current time
#             dt: Time step size
#             dW: Brownian increment with same shape as x
            
#         Returns:
#             Next state with same shape as x
#         """
#         drift = process.drift(x, t)
#         diffusion = process.diffusion(x, t)
        
#         # Compute drift norm for each sample in the batch
#         if x.ndim == 1:
#             # For 1D data (shape [batch_size]), drift norm is just abs value
#             drift_norm = torch.abs(drift)
#         elif x.ndim == 2:
#             # For vector data (shape [batch_size, dim]), compute L2 norm along dim axis
#             drift_norm = torch.linalg.vector_norm(drift, dim=1, keepdim=True)
#         else:
#             # For higher dimensional data (e.g., images [batch_size, C, H, W])
#             # Flatten all dimensions except batch and compute norm
#             batch_size = x.shape[0]
#             drift_flat = drift.reshape(batch_size, -1)
#             drift_norm = torch.linalg.vector_norm(drift_flat, dim=1, keepdim=True)
#             # Reshape back to match original dimensions for broadcasting
#             for _ in range(x.ndim - 2):
#                 drift_norm = drift_norm.unsqueeze(-1)
        
#         # Apply taming factor
#         dt_tensor = torch.as_tensor(dt, dtype=x.dtype, device=x.device)
#         taming_factor = 1.0 / (1.0 + (dt_tensor ** self.alpha) * torch.clamp(drift_norm, min=0.0))
        
#         # Tamed drift
#         drift_tamed = drift * taming_factor
        
#         return x + drift_tamed * dt + diffusion * dW


# class Milstein(Integrator):
#     """
#     Milstein method for scalar SDEs (1D per sample).
#     Includes the additional term for better strong convergence.
    
#     Note: Only implemented for scalar (1D) SDEs. For multi-dimensional SDEs,
#     this requires the diffusion derivative which is problem-specific.
#     """
    
#     def __init__(self, diffusion_derivative: callable = None):
#         """
#         Args:
#             diffusion_derivative: Function that computes ∂σ/∂x (required for Milstein)
#                                 If None, will try to use automatic differentiation.
#         """
#         self.diffusion_derivative = diffusion_derivative
    
#     def step(self, process, x: torch.Tensor, t: float, dt: float, dW: torch.Tensor) -> torch.Tensor:
#         """
#         Milstein integration step (only for 1D SDEs).
        
#         x_{n+1} = x_n + b dt + σ dW + 0.5 σ (∂σ/∂x) (dW² - dt)
#         """
#         if x.ndim > 1 and x.shape[1] > 1:
#             raise NotImplementedError("Milstein method only implemented for scalar (1D) SDEs")
        
#         drift = process.drift(x, t)
#         diffusion = process.diffusion(x, t)
        
#         # Compute diffusion derivative
#         if self.diffusion_derivative is not None:
#             diff_deriv = self.diffusion_derivative(x, t)
#         else:
#             # Use automatic differentiation
#             x_temp = x.requires_grad_(True)
#             sigma = process.diffusion(x_temp, t)
#             diff_deriv = torch.autograd.grad(
#                 sigma.sum(), x_temp, create_graph=False, retain_graph=False
#             )[0]
#             x.requires_grad_(False)
        
#         # Milstein correction term
#         milstein_term = 0.5 * diffusion * diff_deriv * (dW**2 - dt)
        
#         return x + drift * dt + diffusion * dW + milstein_term


# class RungeKutta(Integrator):
#     """
#     Stochastic Runge-Kutta method (strong order 1.0).
#     This is a simplified version for additive noise (diffusion independent of x).
#     """
    
#     def step(self, process, x: torch.Tensor, t: float, dt: float, dW: torch.Tensor) -> torch.Tensor:
#         """
#         Stochastic RK step for SDEs with additive noise.
#         Uses a predictor-corrector approach.
#         """
#         # Predictor step
#         drift = process.drift(x, t)
#         diffusion = process.diffusion(x, t)
#         x_pred = x + drift * dt + diffusion * dW
        
#         # Corrector step (average of drifts)
#         drift_pred = process.drift(x_pred, t + dt)
#         drift_avg = 0.5 * (drift + drift_pred)
        
#         return x + drift_avg * dt + diffusion * dW