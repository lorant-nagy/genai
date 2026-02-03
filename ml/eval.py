#eval.py
import numpy as np
import torch
import torch.nn.functional as F

import ot  # POT
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights

METRIC_KEYS = ['fid', 'kid_mean', 'kid_std', 'w1_pixel', 'w2_pixel', 'w1_emb', 'w2_emb']

_INCEPTION_RES = 299
_INCEPTION_BATCH = 64

_KID_SUBSET_SIZE = 50
_KID_NUM_SUBSETS = 50
_KID_DEGREE = 3
_KID_COEF0 = 1.0

_INCEPTION_MODEL = None  # global cache


def _as_rgb(x: torch.Tensor) -> torch.Tensor:
    """Ensure x is (N,3,H,W). If grayscale (N,1,H,W), repeat channels."""
    if x.ndim != 4:
        raise ValueError(f"Expected (N,C,H,W), got {tuple(x.shape)}")
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    if x.shape[1] == 3:
        return x
    raise ValueError(f"Expected C in {{1,3}}, got C={x.shape[1]}")


def _get_inception(device: torch.device):
    """Create/cached InceptionV3 that outputs 2048-d embeddings (fc replaced by Identity)."""
    global _INCEPTION_MODEL
    if _INCEPTION_MODEL is None:
        weights = Inception_V3_Weights.DEFAULT
        m = inception_v3(weights=weights)
        m.fc = torch.nn.Identity()          # output is 2048-d
        m.eval()
        _INCEPTION_MODEL = m
    return _INCEPTION_MODEL.to(device)


def _inception_embeddings(x01: torch.Tensor) -> np.ndarray:
    """
    x01: float tensor in [0,1], shape (N,C,H,W) with C=1 or 3.
    Returns numpy array (N, 2048).
    """
    device = x01.device
    x = _as_rgb(x01).clamp(0, 1)

    # resize to 299x299
    if x.shape[-1] != _INCEPTION_RES or x.shape[-2] != _INCEPTION_RES:
        x = F.interpolate(x, size=(_INCEPTION_RES, _INCEPTION_RES), mode="bilinear", align_corners=False)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std

    m = _get_inception(device)

    feats = []
    with torch.no_grad():
        for i in range(0, x.shape[0], _INCEPTION_BATCH):
            y = m(x[i:i + _INCEPTION_BATCH])
            feats.append(y.detach().cpu())
    feats = torch.cat(feats, dim=0).numpy().astype(np.float64)
    return feats


def _fid_from_embeddings(feats_r: np.ndarray, feats_g: np.ndarray) -> float:
    __name__ = "fid"
    """Standard FID on embeddings using Gaussian approximation."""
    mu_r = feats_r.mean(axis=0)
    mu_g = feats_g.mean(axis=0)

    cov_r = np.cov(feats_r, rowvar=False)
    cov_g = np.cov(feats_g, rowvar=False)

    covmean = linalg.sqrtm(cov_r @ cov_g)

    # numerical cleanup
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu_r - mu_g
    fid = diff @ diff + np.trace(cov_r + cov_g - 2.0 * covmean)
    return float(fid), __name__


def _poly_mmd2_unbiased(X: np.ndarray, Y: np.ndarray, degree=3, coef0=1.0) -> float:
    """
    Unbiased MMD^2 estimator with polynomial kernel:
        k(x,y) = (gamma * x^T y + coef0)^degree
    gamma = 1/d
    """
    m = X.shape[0]
    if Y.shape[0] != m:
        raise ValueError("KID expects equal number of real and generated features for each subset.")

    d = X.shape[1]
    gamma = 1.0 / d

    Kxx = (gamma * (X @ X.T) + coef0) ** degree
    Kyy = (gamma * (Y @ Y.T) + coef0) ** degree
    Kxy = (gamma * (X @ Y.T) + coef0) ** degree

    # remove diagonals for unbiased estimator
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term_xx = Kxx.sum() / (m * (m - 1))
    term_yy = Kyy.sum() / (m * (m - 1))
    term_xy = Kxy.mean() * 2.0

    return float(term_xx + term_yy - term_xy)


def _kid_from_embeddings(feats_r: np.ndarray, feats_g: np.ndarray) -> (float, float):
    """
    KID = mean and std over multiple random subsets.
    Hardcoded subset size and count.
    """
    n = feats_r.shape[0]
    if feats_g.shape[0] != n:
        raise ValueError("Real and generated must have same number of embeddings for KID here.")

    m = _KID_SUBSET_SIZE
    if n < m:
        raise ValueError(f"Need at least {_KID_SUBSET_SIZE} samples for KID subsets, got {n}.")

    rng = np.random.default_rng(0)
    vals = []
    for _ in range(_KID_NUM_SUBSETS):
        idx_r = rng.choice(n, size=m, replace=False)
        idx_g = rng.choice(n, size=m, replace=False)
        vals.append(_poly_mmd2_unbiased(feats_r[idx_r], feats_g[idx_g], degree=_KID_DEGREE, coef0=_KID_COEF0))

    vals = np.array(vals, dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1))


#v2
# def _w1_w2_pot(X: np.ndarray, Y: np.ndarray) -> (float, float):
#     X = np.asarray(X, dtype=np.float64)
#     Y = np.asarray(Y, dtype=np.float64)

#     if X.shape[0] != Y.shape[0]:
#         raise ValueError("Need same number of samples (minimal helper assumption).")
#     if not (np.isfinite(X).all() and np.isfinite(Y).all()):
#         raise ValueError("Non-finite values in OT inputs (NaN/Inf).")

#     # W1 (earth mover with euclidean cost)
#     W1 = float(ot.solve_sample(X, Y, metric="euclidean").value)

#     # W2 (sqrt of squared-W2 with sqeuclidean cost)
#     W2_sq = float(ot.solve_sample(X, Y, metric="sqeuclidean").value)
#     W2 = float(np.sqrt(max(W2_sq, 0.0)))

#     return W1, W2


def _w1_w2_pot(X: np.ndarray, Y: np.ndarray) -> (float, float):
    """
    Exact OT with uniform weights.
    Returns (W1, W2).
    W2 computed from squared-euclidean cost: W2 = sqrt(emd2(sqeuclidean)).
    """
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError("For this minimal helper we assume same number of real and generated samples.")

    a = np.ones(n) / n
    b = np.ones(n) / n

    M1 = ot.dist(X, Y, metric="euclidean")
    W1 = ot.emd2(a, b, M1)

    M2 = ot.dist(X, Y, metric="sqeuclidean")
    W2_sq = ot.emd2(a, b, M2)
    W2 = np.sqrt(max(W2_sq, 0.0))

    return float(W1), float(W2)


def compute_metrics(real_true: torch.Tensor, gen_true: torch.Tensor) -> dict:
    """
    Minimal wrapper:
      - real_true, gen_true are float tensors in [0,1], shape (N,C,H,W), C=1 or 3
      - returns FID, KID, and POT W1/W2 on pixel and embedding spaces
    """

    results_dict = {}

    if real_true.shape != gen_true.shape:
        raise ValueError(f"Shape mismatch: real {tuple(real_true.shape)} vs gen {tuple(gen_true.shape)}")

    # Inception embeddings
    feats_r = _inception_embeddings(real_true)
    feats_g = _inception_embeddings(gen_true)

    fid = _fid_from_embeddings(feats_r, feats_g)
    kid_mean, kid_std = _kid_from_embeddings(feats_r, feats_g)

    # OT on pixel space (flatten)
    Xr_pix = real_true.detach().cpu().numpy().reshape(real_true.shape[0], -1).astype(np.float64)
    Xg_pix = gen_true.detach().cpu().numpy().reshape(gen_true.shape[0], -1).astype(np.float64)
    w1_pix, w2_pix = _w1_w2_pot(Xr_pix, Xg_pix)

    # OT on embedding space
    w1_emb, w2_emb = _w1_w2_pot(feats_r, feats_g)

    for key in METRIC_KEYS:
        if key == "fid":
            results_dict[key] = fid
        elif key == "kid_mean":
            results_dict[key] = kid_mean
        elif key == "kid_std":
            results_dict[key] = kid_std
        elif key == "w1_pixel":
            results_dict[key] = w1_pix
        elif key == "w2_pixel":
            results_dict[key] = w2_pix
        elif key == "w1_emb":
            results_dict[key] = w1_emb
        elif key == "w2_emb":
            results_dict[key] = w2_emb
        else:
            raise ValueError(f"Unknown metric key: {key}")

    return results_dict