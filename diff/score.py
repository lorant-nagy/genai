# EPS = 1e-12
# import torch

# def vp_ou_score(x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, beta: float) -> torch.Tensor:
#     """
#     VP-OU (constant beta) conditional score.
#     Maps ((B,C,H,W), (B,C,H,W), t) -> (B,C,H,W).
#     """
#     assert x_t.shape == x0.shape and x_t.ndim == 4, "x_t and x0 must be (B,C,H,W)"
#     B = x_t.shape[0]
#     t = t.view(B, 1, 1, 1)
#     a = torch.exp(-0.5 * beta * t)
#     var = torch.clamp(1.0 - a * a, min=1e-12)
#     return -(x_t - a * x0) / var