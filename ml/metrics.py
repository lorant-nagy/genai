"""
ml/metrics.py
-------------
FID, KID (Inception and LeNet), and W1 (exact full-image and sliced pixel OT).

All metric classes accept [0,1] float tensors directly.
Call cache_real(x_real) once, then compute(x_gen) each epoch.
"""

import io
import numpy as np
import torch
import torch.nn as nn
import ot
import urllib.request
from scipy import linalg
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


# ---------------------------------------------------------------------------
# Public OT helper  (full-image exact W1, reusable by external callers)
# ---------------------------------------------------------------------------

def w1(A: torch.Tensor, B: torch.Tensor) -> float:
    """Exact Wasserstein-1 between two sets of flattened images."""
    Xa = A.reshape(A.shape[0], -1).cpu().numpy().astype(np.float64)
    Xb = B.reshape(B.shape[0], -1).cpu().numpy().astype(np.float64)
    return _ot_w1(Xa, Xb)


def _ot_w1(Xa: np.ndarray, Xb: np.ndarray) -> float:
    """Shared OT solver used by both w1() and w1_slice()."""
    n = Xa.shape[0]
    w = np.ones(n) / n
    return float(ot.emd2(w, w, ot.dist(Xa, Xb, metric="euclidean"), numItermax=1_000_000))


# ---------------------------------------------------------------------------
# W1 sliced pixel
# ---------------------------------------------------------------------------

_W1_SLICE_N = 3

def w1_slice(real: torch.Tensor, gen: torch.Tensor, n_slices: int = _W1_SLICE_N) -> float:
    N, C, H, W = real.shape
    v_cols = np.linspace(0, W - 1, n_slices + 2, dtype=int)[1:-1]
    h_rows = np.linspace(0, H - 1, n_slices + 2, dtype=int)[1:-1]

    def extract(x):
        v = x[:, :, :, v_cols].reshape(N, -1)
        h = x[:, :, h_rows, :].reshape(N, -1)
        return np.concatenate([v, h], axis=1).astype(np.float64)

    Xr = extract(real.detach().cpu().numpy())
    Xg = extract(gen.detach().cpu().numpy())
    return _ot_w1(Xr, Xg)


# ---------------------------------------------------------------------------
# Inception helpers
# ---------------------------------------------------------------------------

def _to_uint8(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """[0,1] float (N,C,H,W) -> uint8 RGB on device. C must be 1 or 3."""
    C = x.shape[1]
    if C not in (1, 3):
        raise ValueError(f"Expected C in {{1, 3}}, got C={C}")
    x = x.clamp(0, 1)
    if C == 1:
        x = x.repeat(1, 3, 1, 1)
    return (x * 255).round().clamp(0, 255).to(torch.uint8).to(device)


def _batched_update(metric, imgs: torch.Tensor, real: bool, bs: int) -> None:
    for i in range(0, imgs.shape[0], bs):
        metric.update(imgs[i:i + bs], real=real)


# ---------------------------------------------------------------------------
# InceptionMetrics
# ---------------------------------------------------------------------------

class InceptionMetrics:
    """
    FID and KID via Inception-v3.

    reset_real_features=False makes torchmetrics preserve real features
    across reset() calls, so Inception runs on the real set only once.
    """

    def __init__(self, device: torch.device,
                 fid_feature: int = 2048,
                 kid_feature: int = 2048,
                 kid_subsets: int = 50,
                 kid_subset_size: int = 1000,
                 inception_bs: int = 128):
        self.device       = device
        self.inception_bs = inception_bs

        self._fid = FrechetInceptionDistance(
            feature=fid_feature, reset_real_features=False, normalize=False
        ).to(device)
        self._kid = KernelInceptionDistance(
            feature=kid_feature, subsets=kid_subsets, subset_size=kid_subset_size,
            reset_real_features=False, normalize=False
        ).to(device)

    def cache_real(self, x_real: torch.Tensor) -> None:
        imgs = _to_uint8(x_real, self.device)
        with torch.no_grad():
            _batched_update(self._fid, imgs, real=True, bs=self.inception_bs)
            _batched_update(self._kid, imgs, real=True, bs=self.inception_bs)

    def compute(self, x_gen: torch.Tensor) -> tuple[float, float, float]:
        """Returns (fid, kid_mean, kid_std)."""
        imgs = _to_uint8(x_gen, self.device)
        with torch.no_grad():
            _batched_update(self._fid, imgs, real=False, bs=self.inception_bs)
            _batched_update(self._kid, imgs, real=False, bs=self.inception_bs)
            fid = float(self._fid.compute().cpu())
            km, ks = self._kid.compute()
        self._fid.reset()
        self._kid.reset()
        return fid, float(km.cpu()), float(ks.cpu())


# ---------------------------------------------------------------------------
# LeNet-5  (MNIST domain, 84-d embeddings)
# ---------------------------------------------------------------------------

_LENET_MEAN = 0.1307
_LENET_STD  = 0.3081
_LENET_WEIGHTS_URL = "https://raw.githubusercontent.com/icaros-usc/pyribs/master/tutorials/mnist/mnist_classifier.pth"


def _build_lenet5() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 6, (5, 5)),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(6, 16, (5, 5)),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(256, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10),
        nn.LogSoftmax(dim=1),
    )


def _load_lenet5_embedder(device: torch.device) -> nn.Module:
    """Download weights into memory only — no disk write."""
    print("Downloading LeNet-5 weights into memory...")
    with urllib.request.urlopen(_LENET_WEIGHTS_URL) as resp:
        buf = io.BytesIO(resp.read())
    full_net = _build_lenet5()
    full_net.load_state_dict(torch.load(buf, map_location="cpu"))
    embedder = nn.Sequential(*list(full_net.children())[:-2])
    embedder.eval().requires_grad_(False)
    return embedder.to(device)


def _lenet_encode(embedder: nn.Module, x: torch.Tensor, bs: int) -> np.ndarray:
    feats = []
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            feats.append(embedder(x[i:i + bs]).cpu())
    return torch.cat(feats).numpy().astype(np.float64)


def _fid_from_feats(fa: np.ndarray, fb: np.ndarray) -> float:
    mu_a, mu_b = fa.mean(0), fb.mean(0)
    cov_a, cov_b = np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    covmean = linalg.sqrtm(cov_a @ cov_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_a - mu_b
    return float(diff @ diff + np.trace(cov_a + cov_b - 2.0 * covmean))


def _poly_mmd2_unbiased(X: np.ndarray, Y: np.ndarray, degree=3, coef0=1.0) -> float:
    m, d = X.shape
    gamma = 1.0 / d
    Kxx = (gamma * (X @ X.T) + coef0) ** degree
    Kyy = (gamma * (Y @ Y.T) + coef0) ** degree
    Kxy = (gamma * (X @ Y.T) + coef0) ** degree
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (m*(m-1)) + Kyy.sum() / (m*(m-1)) - 2.0 * Kxy.mean())


def _kid_from_feats(fa, fb, n_subsets, subset_size, seed=None) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.array([
        _poly_mmd2_unbiased(
            fa[rng.choice(fa.shape[0], size=subset_size, replace=False)],
            fb[rng.choice(fb.shape[0], size=subset_size, replace=False)],
        )
        for _ in range(n_subsets)
    ])
    return float(vals.mean()), float(vals.std(ddof=1))


class LeNetMetrics:
    """
    FID and KID using frozen LeNet-5 (84-d penultimate features).
    Only valid for 1-channel 28x28 inputs (MNIST-like).
    Weights are fetched into memory — nothing written to disk.
    """

    def __init__(self, device: torch.device,
                 kid_subsets: int = 50,
                 kid_subset_size: int = 50,
                 bs: int = 256):
        self.device          = device
        self.kid_subsets     = kid_subsets
        self.kid_subset_size = kid_subset_size
        self.bs              = bs
        self.embedder        = _load_lenet5_embedder(device)
        self._feats_real     = None


    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 28 or x.shape[-2] != 28:
            x = torch.nn.functional.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        return ((x.clamp(0, 1) - _LENET_MEAN) / _LENET_STD).to(self.device)

    def cache_real(self, x_real: torch.Tensor) -> None:
        self._feats_real = _lenet_encode(self.embedder, self._preprocess(x_real), self.bs)

    def compute(self, x_gen: torch.Tensor, seed=None) -> tuple[float, float, float]:
        """Returns (fid, kid_mean, kid_std)."""
        fb = _lenet_encode(self.embedder, self._preprocess(x_gen), self.bs)
        fid = _fid_from_feats(self._feats_real, fb)
        km, ks = _kid_from_feats(self._feats_real, fb, self.kid_subsets, self.kid_subset_size, seed)
        return fid, km, ks