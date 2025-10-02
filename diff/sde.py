import torch
from typing import Union, Callable
from diff.sim_core import ItoProcess
from utils.registry import register
from typing import Optional

@register
class VPOU(ItoProcess):
    r"""
    Ornstein–Uhlenbeck (constant β) in R^d:
        dX_t = -1/2 * β * X_t dt + sqrt(β) dW_t

    -- works with batched data:
    - 1D case: x has shape [batch_size]
    - Multi-D case: x has shape [batch_size, dim]
    """

    def __init__(
        self,
        dim: int = 1,
        t0: float = 0.0,
        T: float = 1.0,
        beta: Union[float, torch.Tensor] = 1.0,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(dim=dim, t0=t0, T=T, device=device)

        # parameters
        self.beta = torch.as_tensor(beta, dtype=torch.get_default_dtype(), device=self.device)

        assert torch.all(self.beta > 0), "beta must be positive."
        assert self.beta.ndim in (0, 1), "beta must be scalar or 1D tensor"
        if self.beta.ndim == 1:
            assert self.beta.numel() == dim, f"beta must be scalar or have length {dim}"

        # cache sqrt(beta)
        self._sqrt_beta = torch.sqrt(self.beta)

    def drift(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        b(x,t) = -1/2 * β * x
        """
        if x.ndim == 1:
            return -0.5 * self.beta * x
        else:
            beta = self.beta if self.beta.ndim == 0 else self.beta.unsqueeze(0)
            return -0.5 * beta * x

    def diffusion(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        σ(x,t) = sqrt(β)  (broadcastable to x)
        """
        if x.ndim == 1:
            return self._sqrt_beta
        else:
            if self._sqrt_beta.ndim == 0:
                return self._sqrt_beta
            else:
                return self._sqrt_beta.unsqueeze(0)

    
class OU(ItoProcess):
    r"""
    Ornstein–Uhlenbeck process in R^d:
        dX_t = θ (μ - X_t) dt + σ dW_t
    """

    def __init__(
        self,
        dim: int = 1,
        t0: float = 0.0,
        T: float = 1.0,
        device: str = "cpu",
        theta: Union[float, torch.Tensor] = 1.0,
        mu: Union[float, torch.Tensor] = 0.0,
        sigma: Union[float, torch.Tensor] = 1.0,
    ) -> None:
        super().__init__(dim=dim, t0=t0, T=T, device=device)
        
        # Convert parameters to tensors
        self.theta = torch.as_tensor(theta, dtype=torch.get_default_dtype(), device=self.device)
        self.mu = torch.as_tensor(mu, dtype=torch.get_default_dtype(), device=self.device)
        self.sigma = torch.as_tensor(sigma, dtype=torch.get_default_dtype(), device=self.device)
        
        # Validate shapes
        for param, name in [(self.theta, 'theta'), (self.mu, 'mu'), (self.sigma, 'sigma')]:
            if param.ndim == 1 and param.numel() != dim:
                raise ValueError(f"{name} must be scalar or have length {dim}")
            elif param.ndim > 1:
                raise ValueError(f"{name} must be scalar or 1D tensor")

    def drift(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute drift term θ(μ - x).
        
        Args:
            x: State tensor with shape [batch_size] or [batch_size, dim]
            t: Current time (unused for OU process)
        
        Returns:
            Drift with same shape as x
        """
        # For 1D case (x.shape = [batch_size]), parameters broadcast naturally
        # For multi-D case (x.shape = [batch_size, dim]), ensure proper broadcasting
        if x.ndim == 1:
            # 1D case
            return self.theta * (self.mu - x)
        else:
            # Multi-D case
            theta = self.theta if self.theta.ndim == 0 else self.theta.unsqueeze(0)
            mu = self.mu if self.mu.ndim == 0 else self.mu.unsqueeze(0)
            return theta * (mu - x)

    def diffusion(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute diffusion term σ.
        
        Args:
            x: State tensor with shape [batch_size] or [batch_size, dim]
            t: Current time (unused for OU process)
        
        Returns:
            Diffusion coefficient (constant) broadcastable to x
        """
        if x.ndim == 1:
            # 1D case
            return self.sigma
        else:
            # broadcasting
            if self.sigma.ndim == 0:
                return self.sigma
            else:
                return self.sigma.unsqueeze(0)


class Langevin(ItoProcess):
    r"""
    Overdamped Langevin dynamics:
        dX_t = -(1/γ) ∇U(X_t) dt + √(2/(βγ)) dW_t
    """
    def __init__(
        self,
        dim: int,
        grad_U: Callable[[torch.Tensor], torch.Tensor],
        beta: Union[float, torch.Tensor] = 1.0,
        gamma: Union[float, torch.Tensor] = 1.0,
        t0: float = 0.0,
        T: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(dim=dim, t0=t0, T=T, device=device)
        
        if not callable(grad_U):
            raise TypeError("grad_U must be callable")
        self.grad_U = grad_U
        
        self.beta = torch.as_tensor(beta, dtype=torch.get_default_dtype(), device=self.device)
        self.gamma = torch.as_tensor(gamma, dtype=torch.get_default_dtype(), device=self.device)
        
        if torch.any(self.beta <= 0) or torch.any(self.gamma <= 0):
            raise ValueError("beta and gamma must be positive")
        
        self._sigma = torch.sqrt(2.0 / (self.beta * self.gamma))
        
        # Validate shapes
        for param, name in [(self.beta, 'beta'), (self.gamma, 'gamma'), (self._sigma, 'sigma')]:
            if param.ndim == 1 and param.numel() != dim:
                raise ValueError(f"Derived {name} must be scalar or have length {dim}")
            elif param.ndim > 1:
                raise ValueError(f"{name} must be scalar or 1D tensor")

    def drift(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute drift term -(1/γ)∇U(x).
        
        Args:
            x: State tensor with shape [batch_size] or [batch_size, dim]
            t: Current time (unused for autonomous Langevin)
        
        Returns:
            Drift with same shape as x
        """
        grad = self.grad_U(x)
        
        if grad.shape != x.shape:
            raise ValueError(f"grad_U must return tensor with same shape as x. Got {grad.shape} vs {x.shape}")
        
        if x.ndim == 1:
            # 1D case
            return -grad / self.gamma
        else:
            # Multi-D case
            gamma = self.gamma if self.gamma.ndim == 0 else self.gamma.unsqueeze(0)
            return -grad / gamma

    def diffusion(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute diffusion term √(2/(βγ)).
        
        Args:
            x: State tensor with shape [batch_size] or [batch_size, dim]
            t: Current time (unused)
        
        Returns:
            Diffusion coefficient (constant) broadcastable to x
        """
        if x.ndim == 1:
            # 1D case
            return self._sigma
        else:
            # Multi-D case
            if self._sigma.ndim == 0:
                return self._sigma
            else:
                return self._sigma.unsqueeze(0)


class GeometricBrownianMotion(ItoProcess):
    r"""
    Geometric Brownian Motion:
        dX_t = μ X_t dt + σ X_t dW_t
    
    Common in finance for modeling stock prices.
    """
    
    def __init__(
        self,
        dim: int = 1,
        t0: float = 0.0,
        T: float = 1.0,
        device: str = "cpu",
        mu: Union[float, torch.Tensor] = 0.05,  # drift rate
        sigma: Union[float, torch.Tensor] = 0.2,  # volatility
    ) -> None:
        super().__init__(dim=dim, t0=t0, T=T, device=device)
        
        self.mu = torch.as_tensor(mu, dtype=torch.get_default_dtype(), device=self.device)
        self.sigma = torch.as_tensor(sigma, dtype=torch.get_default_dtype(), device=self.device)
        
        # Validate shapes
        for param, name in [(self.mu, 'mu'), (self.sigma, 'sigma')]:
            if param.ndim == 1 and param.numel() != dim:
                raise ValueError(f"{name} must be scalar or have length {dim}")
            elif param.ndim > 1:
                raise ValueError(f"{name} must be scalar or 1D tensor")
    
    def drift(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Proportional drift μx."""
        if x.ndim == 1:
            return self.mu * x
        else:
            mu = self.mu if self.mu.ndim == 0 else self.mu.unsqueeze(0)
            return mu * x
    
    def diffusion(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Proportional diffusion σx."""
        if x.ndim == 1:
            return self.sigma * x
        else:
            sigma = self.sigma if self.sigma.ndim == 0 else self.sigma.unsqueeze(0)
            return sigma * x