"""
Monte Carlo engine.

The idea: if we know the true data-generating process, we can simulate it R
times, fit OLS every time, and look at the *sampling distribution* of the
estimator directly instead of trusting a formula for it.

For one coefficient this gives us:
    bias          mean(beta_hat) - beta_true
    empirical sd  the actual spread of beta_hat across samples (the truth about precision)
    mean se       what the standard-error formula *claims* the spread is
    se ratio      mean se / empirical sd   (1 = honest, < 1 = overconfident)
    coverage      share of 95% confidence intervals that contain beta_true
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .ols import fit_batch, standard_errors


def run(
    dgp,
    n: int,
    R: int = 5000,
    seed: int = 0,
    coef: int = 1,
    cov_types=("classical",),
    alpha: float = 0.05,
    chunk: int = 5000,
    keep_draws: bool = False,
    **dgp_kwargs,
) -> dict:
    """Simulate `dgp` R times at sample size n and summarise coefficient `coef`."""
    rng = np.random.default_rng(seed)
    betas, ses = [], {c: [] for c in cov_types}
    beta_true = None
    done = 0
    while done < R:
        m = min(chunk, R - done)
        X, y, beta_true = dgp(rng, m, n, **dgp_kwargs)
        fit = fit_batch(X, y)
        betas.append(fit.beta[:, coef])
        for c in cov_types:
            ses[c].append(standard_errors(X, fit, c)[:, coef])
        done += m

    b = np.concatenate(betas)
    truth = beta_true[coef]
    df = n - X.shape[-1]
    crit = stats.t.ppf(1 - alpha / 2, df)
    sd = b.std(ddof=1)

    out = {
        "n": n,
        "R": R,
        "true": truth,
        "mean": b.mean(),
        "bias": b.mean() - truth,
        "emp_sd": sd,
        "rmse": np.sqrt(np.mean((b - truth) ** 2)),
    }
    for c in cov_types:
        s = np.concatenate(ses[c])
        tstat = (b - truth) / s
        out[f"mean_se_{c}"] = s.mean()
        out[f"se_ratio_{c}"] = s.mean() / sd
        out[f"coverage_{c}"] = np.mean(np.abs(tstat) <= crit)
        if keep_draws:
            out[f"t_{c}"] = tstat
            out[f"se_draws_{c}"] = s
    if keep_draws:
        out["draws"] = b
    return out


def mc_standard_error(p: float, R: int) -> float:
    """Simulation noise in an estimated proportion: sqrt(p (1 - p) / R).

    With R = 5000 and p = 0.95 this is about 0.003, so coverage between
    0.944 and 0.956 is statistically indistinguishable from 95%.
    """
    return np.sqrt(p * (1 - p) / R)
