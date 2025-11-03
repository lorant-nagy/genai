from dataclasses import dataclass
import torch


def make_reverse(base_process, score_model, T: float, constant_sigma: bool = True, include_div_a: bool = False, CHW: tuple = None):
    """
    Build the reverse-time SDE Z_τ = X_{T-τ} as a process that runs forward in τ ∈ [0, T].

    Assumptions -- for now:
      - Isotropic diffusion with σ(x,t) that does NOT depend on x (cfg.constant_sigma=True).
      - No divergence term (cfg.include_div_a=False).

    Reverse drift (Song/Anderson/Haussmann–Pardoux, isotropic case):
        b_rev(τ,z) = f(z, T-τ) - g(T-τ)^2 * score(z, T-τ)
    Reverse diffusion:
        σ_rev(τ,z) = g(T-τ)
    """
    if include_div_a:
        raise NotImplementedError(
            "include_div_a=True requires computing ∇·a; not supported while assuming x-independent σ."
        )
    if not constant_sigma:
        raise NotImplementedError(
            "Non-constant-in-x diffusion not supported yet. Set ReverseCfg(constant_sigma=True)."
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
            # τ-forward reverse SDE: f(z, T-τ) - g(T-τ)^2 * score(z, T-τ)
            t = self.T - float(tau)
            f = self.base.drift(z, t)
            g = self.base.diffusion(z, t)
            a = g * g    # using that σ is x-independent
            t_broad = torch.full((z.shape[0],), t, device=z.device, dtype=z.dtype)
            score = self.score_model(z, t_broad)
            return - f + a * score

        def diffusion(self, z, tau):
            # σ, time-flipped
            return self.base.diffusion(z, self.T - float(tau))

    return ReverseProcess()
