from dataclasses import dataclass
import torch


def make_reverse(base_process, score_model, T: float, constant_sigma: bool = True, include_div_a: bool = False, CHW: tuple = None):
    """
    Create reverse-time process for generation/sampling.
    
    Given a forward SDE:
        dX_t = f(X_t, t) dt + σ(X_t, t) dW_t
    
    The reverse-time SDE (with time variable τ = T - t running from 0 to T) is:
        dZ_τ = [-f(Z_τ, T-τ) + a(Z_τ, T-τ) ∇ log p(Z_τ, T-τ)] dτ + σ(Z_τ, T-τ) dW̄_τ
    
    where:
        - Z_τ represents the reverse process starting from Z_0 ~ p_T (noise)
        - a(x,t) = σ(x,t)² is the squared diffusion coefficient
        - ∇ log p(x,t) is the score function (learned by score_model)
        - W̄_τ is a reverse-time Brownian motion
        - T is the terminal time of the forward process
    
    Implementation:
        When integrating backward from t=T to t=0, we use the time variable τ
        that counts forward (τ goes from 0 to T as t goes from T to 0).
        
        The drift is: -f(x, T-τ) + σ²(x, T-τ) * score(x, T-τ)
        
        This is the formula implemented in the drift() method below.
    
    Current assumptions:
        - Isotropic diffusion with σ(x,t) that does NOT depend on x (constant_sigma=True)
        - No divergence term (include_div_a=False)
    
    Args:
        base_process: Forward ItoProcess object defining f(x,t) and σ(x,t)
        score_model: Callable returning ∇ log p(x,t), e.g., trained ScoreNet
        T: Terminal time of forward process
        constant_sigma: Whether diffusion is x-independent (default: True)
        include_div_a: Whether to include divergence of a term (default: False)
        CHW: Channels, Height, Width tuple (unused, kept for compatibility)
    
    Returns:
        ReverseProcess: ItoProcess with reversed drift
    """
    if include_div_a:
        raise NotImplementedError(
            "include_div_a=True requires computing ∇·a; not supported while assuming x-independent σ."
        )
    if not constant_sigma:
        raise NotImplementedError(
            "Non-constant-in-x diffusion not supported yet. Set constant_sigma=True."
        )

    class ReverseProcess(type(base_process)):
        def __init__(self):
            # τ runs from 0 to T
            super().__init__(
                t0=0.0,
                T=float(T),
                device=base_process.device,
            )
            self.base = base_process
            self.score_model = score_model
            self.T = float(T)

        def drift(self, z, tau):
            """
            Reverse-time drift: -f(z, T-τ) + σ²(z, T-τ) * score(z, T-τ)
            
            Args:
                z: State tensor
                tau: Reverse time variable (τ ∈ [0, T])
            
            Returns:
                Drift for reverse SDE
            """
            # Convert τ to forward time
            t = self.T - float(tau)
            
            # Forward drift and diffusion at time t
            f = self.base.drift(z, t)
            g = self.base.diffusion(z, t)
            a = g * g  # σ²(z, t), using that σ is x-independent
            
            # Evaluate score at time t
            t_broad = torch.full((z.shape[0],), t, device=z.device, dtype=z.dtype)
            score = self.score_model(z, t_broad)
            
            # Reverse drift formula
            return -f + a * score
            # return -(- f + a * score) - wrong sign

        def diffusion(self, z, tau):
            """
            Diffusion coefficient is unchanged in reverse time.
            
            Args:
                z: State tensor
                tau: Reverse time variable
            
            Returns:
                σ(z, T-τ)
            """
            return self.base.diffusion(z, self.T - float(tau))

    return ReverseProcess()