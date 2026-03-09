# eval.py
import math
import torch
from ml.metrics import LeNetBackbone

# Populated by the @register_metric decorators when ml.metrics is imported.
# eval.py never imports metric classes directly — it looks them up by name.
from utils.registry import REGISTRY

# Names of all scalar keys that compute() can produce.
# This list drives metrics_evo initialisation in train.py.
METRIC_KEYS = [
    "w1", "w1_slice",
    "fid", "kid_mean", "kid_std",
    "lenet_fid", "lenet_kid_mean", "lenet_kid_std",
    "lenet_entropy",
]

# Metric names that return a (mean, std) pair — handled specially in compute_metrics.
_KID_PAIR_NAMES = {"kid", "lenet_kid"}

_metrics:  dict  = {}   # name -> metric instance
_active:   set   = set()
_device:   torch.device | None = None


def cache_real(real_true: torch.Tensor, device: torch.device, eval_cfg) -> None:
    """
    Instantiate and cache real features for every metric listed in
    eval_cfg.metrics.  eval_cfg.metrics is a list of metric names, e.g.:
      ["w1_slice", "lenet_fid", "lenet_kid", "lenet_entropy"]

    A single LeNetBackbone is created if any lenet_* metric is requested,
    and injected into every LeNet metric that needs it.
    """
    global _metrics, _active, _device
    _device  = device
    _metrics = {}
    _active  = set(eval_cfg.metrics)

    # Build LeNet backbone once if any lenet metric is active.
    lenet_names = {n for n in _active if n.startswith("lenet_")}
    backbone    = LeNetBackbone(device) if lenet_names else None

    lenet_bs          = int(getattr(eval_cfg, "lenet_bs",          256))
    lenet_kid_subsets = int(getattr(eval_cfg, "lenet_kid_subsets",  50))
    lenet_kid_ss      = int(getattr(eval_cfg, "lenet_kid_subset_size", 50))
    inception_bs      = int(getattr(eval_cfg, "inception_bs",      128))
    kid_subsets       = int(getattr(eval_cfg, "kid_subsets",        50))
    kid_ss            = int(getattr(eval_cfg, "kid_subset_size",  1000))

    for name in _active:
        assert name in REGISTRY, \
            f"Unknown metric '{name}'. Did you import ml.metrics?"

        Cls = REGISTRY[name]

        if name == "w1":
            inst = Cls(device=device)
        elif name == "w1_slice":
            inst = Cls(device=device)
        elif name == "fid":
            inst = Cls(device=device, inception_bs=inception_bs)
        elif name == "kid":
            inst = Cls(device=device, kid_subsets=kid_subsets,
                       kid_subset_size=kid_ss, inception_bs=inception_bs)
        elif name in ("lenet_fid", "lenet_entropy"):
            inst = Cls(backbone=backbone, bs=lenet_bs)
        elif name == "lenet_kid":
            inst = Cls(backbone=backbone, bs=lenet_bs,
                       kid_subsets=lenet_kid_subsets, kid_subset_size=lenet_kid_ss)
        else:
            raise AssertionError(f"No construction rule for metric '{name}'")

        inst.cache_real(real_true)
        _metrics[name] = inst


def active_scalar_keys() -> list[str]:
    """
    Returns the scalar result keys that will be produced by compute_metrics,
    in a stable order matching METRIC_KEYS.  Pair metrics (kid, lenet_kid)
    expand to their _mean / _std keys.
    """
    keys = []
    for name in _active:
        if name in _KID_PAIR_NAMES:
            keys.append(f"{name}_mean")
            keys.append(f"{name}_std")
        else:
            keys.append(name)
    order = {k: i for i, k in enumerate(METRIC_KEYS)}
    return sorted(keys, key=lambda k: order.get(k, 999))


def compute_metrics(real_true: torch.Tensor, gen_true: torch.Tensor,
                    kid_seed: int | None = None) -> dict:
    """
    Run every active metric and return a flat dict keyed by METRIC_KEYS.
    Inactive metrics are nan.  KID pair metrics expand to _mean / _std keys.
    """
    results = {}

    for name, inst in _metrics.items():
        val = inst.compute(gen_true, seed=kid_seed)

        if name in _KID_PAIR_NAMES:
            mean, std = val
            results[f"{name}_mean"] = mean
            results[f"{name}_std"]  = std
        else:
            results[name] = val

    return results