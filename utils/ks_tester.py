# ks_tools.py (minimal KS wrapper using scipy.stats.kstest)
import numpy as np
import torch
from scipy.stats import kstest

class KSTester:
    def run(self, data_map):
        """
        data_map: dict name -> [data, cdf]
          - data is (t, X) where X has shape [n_times, 1, n_paths] (d = 1)
          - cdf is a callable cdf(x, t)
        returns: dict name -> (t, pvals) with pvals shape [n_times]
        """
        results = {}
        for name, (data, cdf) in data_map.items():
            t, X = data
            if torch.is_tensor(t):
                t = t.detach().cpu().numpy()
            else:
                t = np.asarray(t)
            if torch.is_tensor(X):
                X = X.detach().cpu().numpy()
            else:
                X = np.asarray(X)

            assert X.ndim == 3 and X.shape[1] == 1, "This KS tester only supports d=1 with X of shape [n_times, 1, n_paths]."

            n_times = X.shape[0]
            pvals = np.zeros(n_times, dtype=float)
            for k in range(n_times):
                tk = float(t[k])
                sample = X[k, 0, :]
                _, p = kstest(sample, lambda x, tt=tk: cdf(x, tt))
                pvals[k] = p

            results[name] = (t, pvals)
        return results
