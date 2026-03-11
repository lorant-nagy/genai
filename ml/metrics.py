"""
ml/metrics.py
-------------
All metric classes accept [0,1] float tensors.
Call cache_real(x_real) once, then compute(x_gen) each epoch.

Each class is registered under its `name` attribute, not its class name,
so the config lists metric names (e.g. "w1_slice", "lenet_fid") directly.
"""

import io
import math
import numpy as np
import torch
import torch.nn as nn
import ot
import urllib.request
from scipy import linalg
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from utils.registry import REGISTRY


# ---------------------------------------------------------------------------
# Registry helper — registers by metric name, not class name
# ---------------------------------------------------------------------------

def register_metric(cls):
    REGISTRY[cls.name] = cls
    return cls


# ---------------------------------------------------------------------------
# OT helpers
# ---------------------------------------------------------------------------

def _ot_w1(Xa: np.ndarray, Xb: np.ndarray) -> float:
    n = Xa.shape[0]
    w = np.ones(n) / n
    return float(ot.emd2(w, w, ot.dist(Xa, Xb, metric="euclidean"), numItermax=1_000_000))


# ---------------------------------------------------------------------------
# Inception helpers
# ---------------------------------------------------------------------------

def _to_uint8(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    C = x.shape[1]
    assert C in (1, 3), f"Expected C in {{1, 3}}, got C={C}"
    x = x.clamp(0, 1)
    if C == 1:
        x = x.repeat(1, 3, 1, 1)
    return (x * 255).round().clamp(0, 255).to(torch.uint8).to(device)


def _batched_update(metric, imgs: torch.Tensor, real: bool, bs: int) -> None:
    for i in range(0, imgs.shape[0], bs):
        metric.update(imgs[i:i + bs], real=real)


# ---------------------------------------------------------------------------
# LeNet backbone (shared across LeNet metrics)
# ---------------------------------------------------------------------------

_LENET_MEAN = 0.1307
_LENET_STD  = 0.3081
_LENET_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/icaros-usc/pyribs/"
    "master/tutorials/mnist/mnist_classifier.pth"
)


def _build_lenet5() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 6, (5, 5), stride=1, padding=0),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(6, 16, (5, 5), stride=1, padding=0),
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


class LeNetBackbone:
    """
    Loads LeNet-5 weights once. Exposes:
      - embedder  : all layers except the last two (outputs 84-d features)
      - classifier: full network              (outputs log-probabilities)
    Both are frozen and in eval mode.
    """

    def __init__(self, device: torch.device):
        self.device = device
        print("Downloading LeNet-5 weights into memory...")
        with urllib.request.urlopen(_LENET_WEIGHTS_URL) as resp:
            buf = io.BytesIO(resp.read())
        full_net = _build_lenet5()
        full_net.load_state_dict(torch.load(buf, map_location="cpu"))
        full_net.eval().requires_grad_(False)

        self.classifier = full_net.to(device)
        self.embedder   = nn.Sequential(*list(full_net.children())[:-2]).to(device)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        if x.shape[-1] != 28 or x.shape[-2] != 28:
            x = torch.nn.functional.interpolate(
                x, size=(28, 28), mode="bilinear", align_corners=False
            )
        return ((x.clamp(0, 1) - _LENET_MEAN) / _LENET_STD).to(self.device)

    def encode(self, x: torch.Tensor, bs: int) -> np.ndarray:
        feats = []
        with torch.no_grad():
            for i in range(0, x.shape[0], bs):
                feats.append(self.embedder(x[i:i + bs]).cpu())
        return torch.cat(feats).numpy().astype(np.float64)

    def classify(self, x: torch.Tensor, bs: int) -> np.ndarray:
        """Returns softmax probabilities [N, 10]."""
        log_probs = []
        with torch.no_grad():
            for i in range(0, x.shape[0], bs):
                log_probs.append(self.classifier(x[i:i + bs]).cpu())
        return torch.cat(log_probs).exp().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# FID / KID math helpers
# ---------------------------------------------------------------------------

def _fid_from_feats(fa: np.ndarray, fb: np.ndarray, eps: float = 1e-6) -> float:
    if fa.shape[0] < 2 or fb.shape[0] < 2:
        return math.nan
    if not np.isfinite(fa).all() or not np.isfinite(fb).all():
        return math.nan
    mu_a, mu_b = fa.mean(axis=0), fb.mean(axis=0)
    cov_a = 0.5 * np.atleast_2d(np.cov(fa, rowvar=False))
    cov_b = 0.5 * np.atleast_2d(np.cov(fb, rowvar=False))
    cov_a = cov_a + cov_a.T
    cov_b = cov_b + cov_b.T
    diff    = mu_a - mu_b
    covmean = linalg.sqrtm(cov_a @ cov_b)
    if not np.isfinite(covmean).all():
        I       = np.eye(cov_a.shape[0], dtype=np.float64)
        covmean = linalg.sqrtm((cov_a + eps * I) @ (cov_b + eps * I))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_sq = diff @ diff + np.trace(cov_a + cov_b - 2.0 * covmean)
    return float(np.sqrt(max(fid_sq, 0.0)))


def _poly_mmd2_unbiased(X: np.ndarray, Y: np.ndarray, degree=3, coef0=1.0) -> float:
    m, d = X.shape
    gamma = 1.0 / d
    Kxx = (gamma * (X @ X.T) + coef0) ** degree
    Kyy = (gamma * (Y @ Y.T) + coef0) ** degree
    Kxy = (gamma * (X @ Y.T) + coef0) ** degree
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (m*(m-1)) + Kyy.sum() / (m*(m-1)) - 2.0 * Kxy.mean())


def _kid_from_feats_gpu(fa, fb, n_subsets, subset_size, seed=None, device=None):
    rng   = np.random.default_rng(seed)
    idx_a = np.stack([rng.choice(fa.shape[0], size=subset_size, replace=False) for _ in range(n_subsets)])
    idx_b = np.stack([rng.choice(fb.shape[0], size=subset_size, replace=False) for _ in range(n_subsets)])
    dev   = device if (device is not None and device.type == "cuda") else torch.device("cpu")
    Xa    = torch.from_numpy(fa[idx_a]).to(dev, dtype=torch.float32)
    Xb    = torch.from_numpy(fb[idx_b]).to(dev, dtype=torch.float32)
    n_sub, m, d = Xa.shape
    gamma = 1.0 / d
    Kxx   = (gamma * torch.bmm(Xa, Xa.transpose(1, 2)) + 1.0) ** 3
    Kyy   = (gamma * torch.bmm(Xb, Xb.transpose(1, 2)) + 1.0) ** 3
    Kxy   = (gamma * torch.bmm(Xa, Xb.transpose(1, 2)) + 1.0) ** 3
    mask  = torch.eye(m, dtype=torch.bool, device=dev).unsqueeze(0)
    Kxx.masked_fill_(mask, 0.0)
    Kyy.masked_fill_(mask, 0.0)
    off  = m * (m - 1)
    vals = (Kxx.view(n_sub, -1).sum(1) / off
            + Kyy.view(n_sub, -1).sum(1) / off
            - 2.0 * Kxy.view(n_sub, -1).mean(1))
    vals_np = np.sqrt(np.maximum(vals.cpu().float().numpy(), 0.0))
    return float(vals_np.mean()), float(vals_np.std(ddof=1))


def _kid_from_feats(fa, fb, n_subsets, subset_size, seed=None, device=None):
    assert fa.shape[0] >= subset_size and fb.shape[0] >= subset_size, \
        f"Need at least {subset_size} samples for KID subsets."
    if device is not None and device.type == "cuda":
        return _kid_from_feats_gpu(fa, fb, n_subsets, subset_size, seed=seed, device=device)
    rng  = np.random.default_rng(seed)
    vals = np.sqrt(np.maximum(np.array([
        _poly_mmd2_unbiased(
            fa[rng.choice(fa.shape[0], size=subset_size, replace=False)],
            fb[rng.choice(fb.shape[0], size=subset_size, replace=False)],
        ) for _ in range(n_subsets)
    ]), 0.0))
    return float(vals.mean()), float(vals.std(ddof=1))


# ---------------------------------------------------------------------------
# W1 metrics
# ---------------------------------------------------------------------------

@register_metric
class W1Metric:
    name = "w1"

    def __init__(self, device: torch.device):
        self.device = device
        self._real: torch.Tensor | None = None

    def cache_real(self, x_real: torch.Tensor) -> None:
        self._real = x_real

    def compute(self, x_gen: torch.Tensor, **_) -> float:
        Xa = self._real.reshape(self._real.shape[0], -1).cpu().numpy().astype(np.float64)
        x_gen_clean = torch.nan_to_num(x_gen, nan=0.0, posinf=1.0, neginf=0.0)
        Xb = x_gen_clean.reshape(x_gen.shape[0], -1).cpu().numpy().astype(np.float64)
        return _ot_w1(Xa, Xb)


@register_metric
class W1SliceMetric:
    name = "w1_slice"

    def __init__(self, device: torch.device, n_projections: int = 256, seed: int | None = None):
        self.device        = device
        self.n_projections = n_projections
        self.seed          = seed
        self._real: torch.Tensor | None = None

    def cache_real(self, x_real: torch.Tensor) -> None:
        self._real = x_real

    def compute(self, x_gen: torch.Tensor, **_) -> float:
        dev = self.device
        Af  = self._real.reshape(self._real.shape[0], -1).to(dev, dtype=torch.float32)
        x_gen_clean = torch.nan_to_num(x_gen, nan=0.0, posinf=1.0, neginf=0.0)
        Bf  = x_gen_clean.reshape(x_gen.shape[0], -1).to(dev, dtype=torch.float32)
        D   = Af.shape[1]
        gen = torch.Generator(device=dev)
        if self.seed is not None:
            gen.manual_seed(self.seed)
        dirs = torch.randn(D, self.n_projections, device=dev, generator=gen)
        dirs = dirs / dirs.norm(dim=0, keepdim=True)
        pa   = (Af @ dirs).sort(dim=0).values
        pb   = (Bf @ dirs).sort(dim=0).values
        return float((pa - pb).abs().mean())


# ---------------------------------------------------------------------------
# Inception metrics
# ---------------------------------------------------------------------------

@register_metric
class InceptionFIDMetric:
    name = "fid"

    def __init__(self, device: torch.device, feature: int = 2048, inception_bs: int = 128):
        self.device = device
        self.bs     = inception_bs
        self._fid   = FrechetInceptionDistance(
            feature=feature, reset_real_features=False, normalize=False
        ).to(device)

    def cache_real(self, x_real: torch.Tensor) -> None:
        with torch.no_grad():
            _batched_update(self._fid, _to_uint8(x_real, self.device), real=True, bs=self.bs)

    def compute(self, x_gen: torch.Tensor, **_) -> float:
        with torch.no_grad():
            _batched_update(self._fid, _to_uint8(x_gen, self.device), real=False, bs=self.bs)
            val = float(self._fid.compute().cpu())
            self._fid.reset()
        return val


@register_metric
class InceptionKIDMetric:
    name = "kid"

    def __init__(self, device: torch.device, feature: int = 2048,
                 kid_subsets: int = 50, kid_subset_size: int = 1000,
                 inception_bs: int = 128):
        self.device = device
        self.bs     = inception_bs
        self._kid   = KernelInceptionDistance(
            feature=feature, subsets=kid_subsets, subset_size=kid_subset_size,
            reset_real_features=False, normalize=False
        ).to(device)

    def cache_real(self, x_real: torch.Tensor) -> None:
        with torch.no_grad():
            _batched_update(self._kid, _to_uint8(x_real, self.device), real=True, bs=self.bs)

    def compute(self, x_gen: torch.Tensor, **_) -> tuple[float, float]:
        with torch.no_grad():
            _batched_update(self._kid, _to_uint8(x_gen, self.device), real=False, bs=self.bs)
            km, ks = self._kid.compute()
            self._kid.reset()
        return float(km.cpu().sqrt()), float(ks.cpu().sqrt())


# ---------------------------------------------------------------------------
# LeNet metrics  (require a LeNetBackbone injected at construction)
# ---------------------------------------------------------------------------

@register_metric
class LeNetFIDMetric:
    name = "lenet_fid"

    def __init__(self, backbone: LeNetBackbone, bs: int = 256):
        self.backbone     = backbone
        self.bs           = bs
        self._feats_real: np.ndarray | None = None

    def cache_real(self, x_real: torch.Tensor) -> None:
        self._feats_real = self.backbone.encode(self.backbone.preprocess(x_real), self.bs)

    def compute(self, x_gen: torch.Tensor, **_) -> float:
        fb = self.backbone.encode(self.backbone.preprocess(x_gen), self.bs)
        return _fid_from_feats(self._feats_real, fb)


@register_metric
class LeNetKIDMetric:
    name = "lenet_kid"

    def __init__(self, backbone: LeNetBackbone, bs: int = 256,
                 kid_subsets: int = 50, kid_subset_size: int = 50):
        self.backbone     = backbone
        self.bs           = bs
        self.kid_subsets     = kid_subsets
        self.kid_subset_size = kid_subset_size
        self._feats_real: np.ndarray | None = None

    def cache_real(self, x_real: torch.Tensor) -> None:
        self._feats_real = self.backbone.encode(self.backbone.preprocess(x_real), self.bs)

    def compute(self, x_gen: torch.Tensor, seed=None, **_) -> tuple[float, float]:
        fb = self.backbone.encode(self.backbone.preprocess(x_gen), self.bs)
        return _kid_from_feats(
            self._feats_real, fb,
            n_subsets=self.kid_subsets,
            subset_size=self.kid_subset_size,
            seed=seed,
            device=self.backbone.device,
        )


@register_metric
class LeNetEntropyMetric:
    """
    Shannon entropy of the marginal class distribution predicted by LeNet.

    For each generated image the classifier produces a probability vector
    over 10 classes. We average those vectors across the batch to get the
    marginal distribution p, then return H(p) = -sum(p * log(p)).

    Interpretation:
      max entropy  log(10) ≈ 2.30  — generator covers all classes uniformly
      near 0                        — generator has collapsed to one class

    No cache_real needed — this is a self-contained measure on x_gen only.
    """
    name = "lenet_entropy"

    def __init__(self, backbone: LeNetBackbone, bs: int = 256):
        self.backbone = backbone
        self.bs       = bs

    def cache_real(self, x_real: torch.Tensor) -> None:
        pass  # not used

    def compute(self, x_gen: torch.Tensor, **_) -> float:
        probs = self.backbone.classify(self.backbone.preprocess(x_gen), self.bs)
        marginal = probs.mean(axis=0)                          # [10]
        marginal = np.clip(marginal, 1e-12, None)
        return float(-np.sum(marginal * np.log(marginal)))