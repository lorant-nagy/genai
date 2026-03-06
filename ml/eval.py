# eval.py
import math
import torch
from ml.metrics import InceptionMetrics, LeNetMetrics, w1, w1_slice

METRIC_KEYS = ['w1', 'w1_slice', 'fid', 'kid_mean', 'kid_std', 'lenet_fid', 'lenet_kid_mean', 'lenet_kid_std']

_inception: InceptionMetrics | None = None
_lenet:     LeNetMetrics     | None = None
_use_lenet:         bool = False
_use_inception_fid: bool = True
_use_inception_kid: bool = True
_use_w1:            bool = True
_use_w1_slice:      bool = True
_use_stds:          bool = True
_device:            torch.device | None = None


def cache_real(real_true: torch.Tensor, device: torch.device, eval_cfg) -> None:
    """
    Call once after real_true is collected.
    eval_cfg is config.eval (a Cfg object with the metric params).

    Optional flags (all default True if absent from config):
      use_inception_fid : compute Inception FID
      use_inception_kid : compute Inception KID
      use_w1            : compute exact W1 (expensive; w1_slice always runs)
      use_stds          : include kid_std and lenet_kid_std in results
      use_lenet         : compute LeNet FID/KID
    """
    global _inception, _lenet, _use_lenet, _device
    global _use_inception_fid, _use_inception_kid, _use_w1, _use_w1_slice, _use_stds

    _device = device

    _use_inception_fid = bool(getattr(eval_cfg, 'use_inception_fid', True))
    _use_inception_kid = bool(getattr(eval_cfg, 'use_inception_kid', True))
    _use_w1            = bool(getattr(eval_cfg, 'use_w1',            True))
    _use_w1_slice      = bool(getattr(eval_cfg, 'use_w1_slice',      True))
    _use_stds          = bool(getattr(eval_cfg, 'use_stds',          True))

    use_inception = _use_inception_fid or _use_inception_kid
    if use_inception:
        _inception = InceptionMetrics(
            device          = device,
            fid_feature     = 2048,
            kid_feature     = 2048,
            kid_subsets     = eval_cfg.kid_subsets,
            kid_subset_size = eval_cfg.kid_subset_size,
            inception_bs    = eval_cfg.inception_bs,
            use_fid         = _use_inception_fid,
            use_kid         = _use_inception_kid,
        )
        _inception.cache_real(real_true)
    else:
        _inception = None

    _use_lenet = (
        bool(getattr(eval_cfg, 'use_lenet', False))
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

    # W1 exact (optional) and w1_slice (always)
    results['w1']       = w1(real_true, gen_true) if _use_w1 else math.nan
    results['w1_slice'] = w1_slice(real_true, gen_true, device=_device) if _use_w1_slice else math.nan

    # Inception FID + KID
    if _inception is not None:
        fid, km, ks = _inception.compute(gen_true)
    else:
        fid, km, ks = math.nan, math.nan, math.nan
    results['fid']      = fid
    results['kid_mean'] = km
    results['kid_std']  = ks if _use_stds else math.nan

    # LeNet FID + KID
    if _use_lenet:
        lfid, lkm, lks = _lenet.compute(gen_true, seed=kid_seed)
        results['lenet_fid']      = lfid
        results['lenet_kid_mean'] = lkm
        results['lenet_kid_std']  = lks if _use_stds else math.nan
    else:
        results['lenet_fid']      = math.nan
        results['lenet_kid_mean'] = math.nan
        results['lenet_kid_std']  = math.nan

    return results