# eval.py
import numpy as np
import torch
import torch.nn.functional as F

import ot  # POT
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights

# METRIC_KEYS = ['fid', 'kid_mean', 'kid_std' , 'w1_emb', 'w1_slice']
METRIC_KEYS = ['w1_slice']
# METRIC_KEYS = ['fid', 'kid_mean', 'kid_std', 'w1_pixel', 'w2_pixel', 'w1_emb', 'w2_emb']

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
    return float(fid)


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


def _kid_from_embeddings(feats_r: np.ndarray, feats_g: np.ndarray, seed: np.random.SeedSequence | int | None = None) -> (float, float):
    """
    KID = mean and std over multiple random subsets.

    If seed is provided, the subset sampling is deterministic.
    """
    n = feats_r.shape[0]
    if feats_g.shape[0] != n:
        raise ValueError("Real and generated must have same number of embeddings for KID here.")

    m = _KID_SUBSET_SIZE
    if n < m:
        raise ValueError(f"Need at least {_KID_SUBSET_SIZE} samples for KID subsets, got {n}.")

    rng = np.random.default_rng(seed)

    vals = []
    for _ in range(_KID_NUM_SUBSETS):
        idx_r = rng.choice(n, size=m, replace=False)
        idx_g = rng.choice(n, size=m, replace=False)
        vals.append(_poly_mmd2_unbiased(feats_r[idx_r], feats_g[idx_g], degree=_KID_DEGREE, coef0=_KID_COEF0))

    vals = np.array(vals, dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1))


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

_W1_SLICE_N = 3

def _w1_sliced_pixel(real: torch.Tensor, gen: torch.Tensor, n_slices: int = _W1_SLICE_N) -> float:
    """
    W1 on concatenated horizontal + vertical pixel slices.
    
    real, gen: (N, C, H, W) tensors in [0,1].
    Returns scalar W1.
    """
    N, C, H, W = real.shape
    
    # Slice positions: evenly spaced, excluding edges
    v_cols = np.linspace(0, W - 1, n_slices + 2, dtype=int)[1:-1]  # vertical slices
    h_rows = np.linspace(0, H - 1, n_slices + 2, dtype=int)[1:-1]  # horizontal slices
    
    def extract(x):
        # x: (N, C, H, W)
        v = x[:, :, :, v_cols]          # (N, C, H, n_slices)
        h = x[:, :, h_rows, :]          # (N, C, n_slices, W)
        v_flat = v.reshape(N, -1)       # (N, C*H*n_slices)
        h_flat = h.reshape(N, -1)       # (N, C*n_slices*W)
        return np.concatenate([v_flat, h_flat], axis=1)  # (N, C*(H+W)*n_slices)
    
    Xr = extract(real.detach().cpu().numpy()).astype(np.float64)
    Xg = extract(gen.detach().cpu().numpy()).astype(np.float64)
    
    a = np.ones(N) / N
    b = np.ones(N) / N
    M = ot.dist(Xr, Xg, metric="euclidean")
    return float(ot.emd2(a, b, M))


def compute_metrics(real_true, gen_true, kid_seed: int | None = None):
    """
    Compute metrics between real_true and gen_true.

    kid_seed:
      - None  -> KID subset sampling is random (current behavior).
      - int   -> deterministic KID subset sampling for stable eval/profiling.
    """
    keys = set(METRIC_KEYS)
    results = {}

    need_emb = any(k in keys for k in ["fid", "kid_mean", "kid_std", "w1_emb", "w2_emb"])
    need_pix = any(k in keys for k in ["w1_pixel", "w2_pixel"])

    feats_r = feats_g = None
    if need_emb:
        feats_r = _inception_embeddings(real_true)
        feats_g = _inception_embeddings(gen_true)

    if "fid" in keys:
        results["fid"] = _fid_from_embeddings(feats_r, feats_g)

    if ("kid_mean" in keys) or ("kid_std" in keys):
        km, ks = _kid_from_embeddings(feats_r, feats_g, seed=kid_seed)
        if "kid_mean" in keys: results["kid_mean"] = km
        if "kid_std"  in keys: results["kid_std"]  = ks

    if need_pix:
        Xr = real_true.detach().cpu().numpy().reshape(real_true.shape[0], -1).astype(np.float64)
        Xg = gen_true.detach().cpu().numpy().reshape(gen_true.shape[0], -1).astype(np.float64)
        w1p, w2p = _w1_w2_pot(Xr, Xg)
        if "w1_pixel" in keys: results["w1_pixel"] = w1p
        if "w2_pixel" in keys: results["w2_pixel"] = w2p

    if ("w1_emb" in keys) or ("w2_emb" in keys):
        w1e, w2e = _w1_w2_pot(feats_r, feats_g)
        if "w1_emb" in keys: results["w1_emb"] = w1e
        if "w2_emb" in keys: results["w2_emb"] = w2e

    if "w1_slice" in keys:
        results["w1_slice"] = _w1_sliced_pixel(real_true, gen_true)

    return results
