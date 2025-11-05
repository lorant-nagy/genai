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
        return local_ou_score_nonlinear(
            x_t,
            x0,
            t,
            alpha=self.alpha,
            c_alpha=self.c_alpha,
            c0=self.c_0,
            sigma=self.sigma,
            steps=steps,
        )



def local_ou_score_nonlinear(
    x_t: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor,
    *,
    alpha: float = 3.0,
    c_alpha: float = 1.0,
    c0: float = 0.5,
    sigma: float = 1.0,
    steps: int = 1,
):
    """
    Local-OU approximation of the conditional score ∇_{x_t} log p(x_t | x0, t)
    for the 1D nonlinear Langevin SDE (applied elementwise over tensors):

        dX_t = - (c_alpha * |X_t|^alpha * sign(X_t) + c0 * X_t) dt + sigma dW_t

    Strategy:
      - Linearise the drift around the current anchor (initially x0):
            b(x) ≈ b(x_anchor) + J_anchor * (x - x_anchor),
        where  b(x) = - (c_alpha |x|^alpha sign(x) + c0 x),
              J_anchor = b'(x_anchor) = - (c_alpha * alpha * |x_anchor|^(alpha-1) + c0).
      - This yields a 1D OU SDE with *constant* coefficients over a short step Δt,
        so X_{t+Δt} | X_t is Gaussian with known mean/variance.
      - Optionally re-linearise `steps` times along the evolving mean to improve accuracy
        for larger t. The final law is approximated as N(mu, var), so the score is:
            score(x_t | x0, t) = - (x_t - mu) / var.

    Shapes:
      x_t, x0: (B, C, H, W)
      t:      (B,) or (B,1,1,1)

    Args:
      alpha, c_alpha, c0, sigma: SDE parameters.
      steps: number of re-linearisation substeps (>=1). steps=1 reproduces the
             single linearisation around x0 (good for small t).

    Returns:
      Tensor (B, C, H, W): approximate conditional score at x_t.
    """
    assert x_t.shape == x0.shape and x_t.ndim == 4, "x_t and x0 must be (B,C,H,W)"
    B = x_t.shape[0]
    device = x_t.device
    dtype = x_t.dtype

    # Ensure t is (B,1,1,1)
    t = t.view(B, 1, 1, 1).to(device=device, dtype=dtype)

    # Helper: one OU step over dt given anchor.
    def ou_step(anchor, dt):
        # b(anchor) = - (c_alpha |anchor|^alpha sign(anchor) + c0 * anchor)
        abs_a = torch.abs(anchor)
        sign_a = torch.sign(anchor)
        b_a = - (c_alpha * abs_a.pow(alpha) * sign_a + c0 * anchor)

        # J = b'(anchor) = - (c_alpha * alpha * |anchor|^(alpha-1) + c0)
        # Note: for alpha > 1, derivative at 0 is 0 for the |x|^alpha term.
        pow_term = torch.where(abs_a > 0, abs_a.pow(alpha - 1.0), torch.zeros_like(abs_a))
        J = - (c_alpha * alpha * pow_term + c0)

        # Stable φ = (e^{J dt} - 1) / J and ψ = (e^{2J dt} - 1) / (2J)
        Jdt = J * dt
        # torch.expm1 for numerical stability when Jdt ~ 0
        expm1_Jdt = torch.expm1(Jdt)
        expm1_2Jdt = torch.expm1(2.0 * Jdt)

        # Avoid 0/0 by branching when |J| is tiny; use limits φ≈dt, ψ≈dt
        small = (torch.abs(J) < 1e-8)
        invJ = torch.where(small, torch.zeros_like(J), 1.0 / (J + (~small) * 0.0 + small * 1.0))  # safe inverse
        phi = torch.where(small, dt, expm1_Jdt * invJ)
        psi = torch.where(small, dt, 0.5 * expm1_2Jdt * invJ)

        # Mean/variance update for OU step with drift J x + c,
        # where c = b(anchor) - J * anchor. Using the identity:
        #   mu_next = anchor + φ * b(anchor)
        #   var_innov = σ^2 * ψ
        mu_next = anchor + phi * b_a
        var_innov = (sigma * sigma) * psi

        # Also need the linear multiplier A = e^{J dt} to propagate existing variance.
        A = torch.exp(Jdt)

        return mu_next, var_innov, A

    # Initialise mean and variance of X given X0=x0 at time 0.
    mu = x0
    var = torch.zeros_like(x0)

    # Split time into equal substeps per batch element (broadcast ok).
    dt = t / float(steps)

    # Re-linearise `steps` times along the evolving mean.
    for _ in range(steps):
        mu_next, var_innov, A = ou_step(mu, dt)
        # Propagate variance: var_{k+1} = A^2 var_k + var_innov
        var = (A * A) * var + var_innov
        mu = mu_next

    var = torch.clamp(var, min=1e-12)
    return - (x_t - mu) / var


def local_ou_mean_var(
    x0: torch.Tensor,
    t: torch.Tensor,
    *,
    alpha: float = 3.0,
    c_alpha: float = 1.0,
    c0: float = 0.5,
    sigma: float = 1.0,
    steps: int = 1,
):
    """
    Utility: returns (mu, var) of the Gaussian approximation used by
    local_ou_score_nonlinear for debugging/inspection.
    """
    B = x0.shape[0]
    device = x0.device
    dtype = x0.dtype
    t = t.view(B, 1, 1, 1).to(device=device, dtype=dtype)

    def ou_step(anchor, dt):
        abs_a = torch.abs(anchor)
        sign_a = torch.sign(anchor)
        b_a = - (c_alpha * abs_a.pow(alpha) * sign_a + c0 * anchor)
        pow_term = torch.where(abs_a > 0, abs_a.pow(alpha - 1.0), torch.zeros_like(abs_a))
        J = - (c_alpha * alpha * pow_term + c0)

        Jdt = J * dt
        expm1_Jdt = torch.expm1(Jdt)
        expm1_2Jdt = torch.expm1(2.0 * Jdt)
        small = (torch.abs(J) < 1e-8)
        invJ = torch.where(small, torch.zeros_like(J), 1.0 / (J + (~small) * 0.0 + small * 1.0))
        phi = torch.where(small, dt, expm1_Jdt * invJ)
        psi = torch.where(small, dt, 0.5 * expm1_2Jdt * invJ)
        mu_next = anchor + phi * b_a
        var_innov = (sigma * sigma) * psi
        A = torch.exp(Jdt)
        return mu_next, var_innov, A

    mu = x0
    var = torch.zeros_like(x0)
    dt = t / float(steps)
    for _ in range(steps):
        mu_next, var_innov, A = ou_step(mu, dt)
        var = (A * A) * var + var_innov
        mu = mu_next
    var = torch.clamp(var, min=1e-12)
    return mu, var