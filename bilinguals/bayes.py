"""Bayes factors for lesion-load / score correlations in each group
(replaces BayesNullCorr and the external jzs_corbf)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate, special


def jzs_bf10(r: float, n: int) -> float:
    """JZS Bayes factor for a Pearson correlation (Wetzels & Wagenmakers, 2012):

        BF10 = E_g[(1+g)^((n-2)/2) * (1+(1-r^2) g)^(-(n-1)/2)],  g ~ InvGamma(1/2, n/2)

    BF10 > 1 favours a correlation; < 1 favours none.
    """
    if n < 3 or not np.isfinite(r):
        return float("nan")
    r2 = min(r * r, 1 - 1e-12)
    a, b = (n - 2) / 2.0, (n - 1) / 2.0
    log_norm = 0.5 * np.log(n / 2.0) - special.gammaln(0.5)

    def log_f(u):  # integrate over u = log g
        g = np.exp(u)
        return a * np.log1p(g) - b * np.log1p((1 - r2) * g) - 0.5 * u - n / (2 * g)

    grid = np.linspace(-15, 25, 4001)
    with np.errstate(divide="ignore", over="ignore"):
        vals = log_f(grid)
    peak = grid[np.argmax(vals)]
    shift = vals.max()
    # Below the peak the integrand vanishes like exp(-n/2g); above it, like 1/g.
    # These limits leave out less than 1e-15 of the mass.
    total = 0.0
    for lo, hi in ((peak - 40, peak), (peak, peak + 45)):
        piece, _ = integrate.quad(lambda u: np.exp(log_f(u) - shift), lo, hi, limit=400)
        total += piece
    return float(np.exp(log_norm + shift) * total)


def correlation_table(X: pd.DataFrame, scores: pd.DataFrame, l1, min_lesioned=10) -> pd.DataFrame:
    """For every lesion-load column x score: r and BF10 in each group.

    A pair is only tested when at least ``min_lesioned`` patients in each group
    have a non-zero lesion load (as BayesNullCorr). Correlations use all
    patients with both values, including zero loads.
    """
    l1 = np.asarray(l1, bool)
    rows = []
    for region in X.columns:
        load = X[region].to_numpy(float)
        for task in scores.columns:
            y = scores[task].to_numpy(float)
            ok = ~np.isnan(load) & ~np.isnan(y)
            if min(np.sum(ok & (load > 0) & l1), np.sum(ok & (load > 0) & ~l1)) < min_lesioned:
                continue
            row = {"region": region, "task": task}
            for name, g in (("l1", l1), ("l2", ~l1)):
                sel = ok & g
                r = np.corrcoef(load[sel], y[sel])[0, 1] if np.ptp(load[sel]) > 0 else np.nan
                row.update({f"{name}_n": int(sel.sum()), f"{name}_r": r, f"{name}_bf10": jzs_bf10(r, int(sel.sum()))})
            rows.append(row)
    return pd.DataFrame(rows)
