# eval.py
import math
import torch
from ml.metrics import InceptionMetrics, LeNetMetrics, w1, w1_slice

METRIC_KEYS = ['w1', 'w1_slice', 'fid', 'kid_mean', 'kid_std', 'lenet_fid', 'lenet_kid_mean', 'lenet_kid_std']

_inception: InceptionMetrics | None = None
_lenet:     LeNetMetrics     | None = None
_use_lenet: bool = False


def cache_real(real_true: torch.Tensor, device: torch.device, eval_cfg) -> None:
    """
    Call once after real_true is collected.
    eval_cfg is config.eval (a Cfg object with the metric params).
    """
    global _inception, _lenet, _use_lenet

    _inception = InceptionMetrics(
        device        = device,
        kid_subsets   = eval_cfg.kid_subsets,
        kid_subset_size = eval_cfg.kid_subset_size,
        inception_bs  = eval_cfg.inception_bs,
    )
    _inception.cache_real(real_true)

    _use_lenet = (
        eval_cfg.use_lenet
        and real_true.shape[1] == 1
    )
    if _use_lenet:
        _lenet = LeNetMetrics(
            device          = device,
            kid_subsets     = eval_cfg.lenet_kid_subsets,
            kid_subset_size = eval_cfg.lenet_kid_subset_size,
            bs              = eval_cfg.lenet_bs,
        )
        _lenet.cache_real(real_true)


def compute_metrics(real_true: torch.Tensor, gen_true: torch.Tensor,
                    kid_seed: int | None = None) -> dict:
    results = {}

    fid, km, ks = _inception.compute(gen_true)
    results['fid']      = fid
    results['kid_mean'] = km
    results['kid_std']  = ks
    results['w1']       = w1(real_true, gen_true)
    results['w1_slice'] = w1_slice(real_true, gen_true)

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