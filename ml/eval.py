# eval.py
import math
import torch
from ml.metrics import InceptionMetrics, LeNetMetrics, w1_slice

METRIC_KEYS = ['w1_slice', 'fid', 'kid_mean', 'kid_std', 'lenet_fid', 'lenet_kid_mean', 'lenet_kid_std']

_inception: InceptionMetrics | None = None
_lenet: LeNetMetrics | None = None
_use_lenet: bool = False


def cache_real(real_true: torch.Tensor, device: torch.device) -> None:
    """Call once after real_true is collected. Instantiates and caches metric objects."""
    global _inception, _lenet, _use_lenet

    _inception = InceptionMetrics(device=device)
    _inception.cache_real(real_true)

    _use_lenet = (real_true.shape[1] == 1 and real_true.shape[2] == 28 and real_true.shape[3] == 28)
    if _use_lenet:
        _lenet = LeNetMetrics(device=device)
        _lenet.cache_real(real_true)


def compute_metrics(real_true: torch.Tensor, gen_true: torch.Tensor,
                    kid_seed: int | None = None) -> dict:
    results = {}

    results['w1_slice'] = w1_slice(real_true, gen_true)

    fid, km, ks = _inception.compute(gen_true)
    results['fid']      = fid
    results['kid_mean'] = km
    results['kid_std']  = ks

    if _use_lenet:
        lfid, lkm, lks = _lenet.compute(gen_true, seed=kid_seed)
        results['lenet_fid']      = lfid
        results['lenet_kid_mean'] = lkm
        results['lenet_kid_std']  = lks
    else:
        results['lenet_fid']      = math.nan
        results['lenet_kid_mean'] = math.nan
        results['lenet_kid_std']  = math.nan

    return results