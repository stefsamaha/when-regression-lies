"""
Experiment 4: repairing inference.

4a. Heteroskedasticity: classical vs HC0, HC1, HC2, HC3 across sample sizes.
    HC0 is consistent but biased downward in small samples because residuals
    are smaller than errors on high-leverage points (Var(e_i) = sigma^2 (1 - h_ii)).
    HC2/HC3 undo this by inflating those residuals.
4b. Efficiency: under heteroskedasticity OLS is no longer BLUE. Weighted least
    squares with the true variance function is (Aitken's theorem).
4c. Autocorrelation: how classical and Newey-West SEs degrade as persistence grows.
4d. Joint hypotheses: the classical F-test vs a robust Wald test under heteroskedasticity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regression_lies import dgp, montecarlo, fit_batch, wald_test
from regression_lies.ols import covariance
from regression_lies.plotting import (setup, save, TAB_DIR, BLUE, ORANGE, AQUA, YELLOW,
                                      MAGENTA, INK2, GRID)

HC_TYPES = ("classical", "HC0", "HC1", "HC2", "HC3")
NS = [10, 20, 40, 80, 160, 320, 1000]


def hc_comparison(R, seed=2000):
    rows = []
    for i, n in enumerate(NS):
        res = montecarlo.run(dgp.heteroskedastic, n=n, R=R, seed=seed + i, cov_types=HC_TYPES)
        rows.append({"n": n, **{c: res[f"coverage_{c}"] for c in HC_TYPES},
                     **{f"se_ratio_{c}": res[f"se_ratio_{c}"] for c in HC_TYPES}})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.axhspan(0.94, 0.96, color=GRID, zorder=0)
    ax.axhline(0.95, color=INK2, ls="--", lw=1)
    for c, col in zip(HC_TYPES, [ORANGE, YELLOW, MAGENTA, AQUA, BLUE]):
        ax.plot(df.n, df[c], "o-", color=col, label=c if c != "classical" else "Classical", ms=6)
    ax.set_xscale("log")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("95% CI coverage")
    ax.set_title("Heteroskedasticity-robust standard errors\nHC3 is closest to 95% in small samples")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    save(fig, "04a_hc_comparison.png")
    return df


def wls_efficiency(R, seed=2100, n=100):
    """OLS vs WLS (weights 1/Var(eps|x), assumed known) under heteroskedasticity."""
    rng = np.random.default_rng(seed)
    X, y, beta = dgp.heteroskedastic(rng, R, n)
    b_ols = fit_batch(X, y).beta[:, 1]
    w = 1 / np.sqrt(0.1 + X[..., 1] ** 2)          # divide each row by its error sd
    b_wls = fit_batch(X * w[..., None], y * w).beta[:, 1]
    return pd.DataFrame({
        "estimator": ["OLS", "WLS (true weights)"],
        "mean": [b_ols.mean(), b_wls.mean()],
        "sd": [b_ols.std(ddof=1), b_wls.std(ddof=1)],
    })


def hac_vs_persistence(R, seed=2200, n=200):
    rows = []
    for i, rho in enumerate([0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]):
        res = montecarlo.run(dgp.autocorrelated, n=n, R=R, seed=seed + i,
                             cov_types=("classical", "HC3", "HAC"), rho_x=rho, rho_e=rho)
        inflation = (1 + rho**2) / (1 - rho**2)      # Var(b1) / classical Var(b1), large-n approximation
        rows.append({"rho": rho, "theory_variance_inflation": inflation,
                     "mc_variance_inflation": 1 / res["se_ratio_classical"] ** 2,
                     **{f"coverage_{c}": res[f"coverage_{c}"] for c in ("classical", "HC3", "HAC")}})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(df.rho, df.theory_variance_inflation, color=INK2, ls="--",
            label=r"theory $(1+\rho^2)/(1-\rho^2)$")
    ax.plot(df.rho, df.mc_variance_inflation, "o", color=ORANGE, ms=8, label="Monte Carlo")
    ax.set_yscale("log")
    ax.set_xlabel(r"persistence $\rho$ of both x and the error")
    ax.set_ylabel("true Var(slope) / classical estimate")
    ax.set_title(f"(a) How much the classical SE understates variance (n = {n})")
    ax.legend()

    ax = axes[1]
    ax.axhline(0.95, color=INK2, ls="--", lw=1)
    for c, col, lab in [("classical", ORANGE, "Classical"), ("HC3", BLUE, "HC3"),
                        ("HAC", AQUA, "Newey-West HAC")]:
        ax.plot(df.rho, df[f"coverage_{c}"], "o-", color=col, label=lab)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"persistence $\rho$")
    ax.set_ylabel("95% CI coverage")
    ax.set_title("(b) HC3 does nothing for autocorrelation;\nHAC helps but under-covers when persistence is high")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "04b_autocorrelation_hac.png")
    return df


def joint_test_size(R, seed=2300):
    """H0: b1 = b2 = 0 (true) with heteroskedastic errors driven by both regressors."""
    rows = []
    rng = np.random.default_rng(seed)
    for n in [25, 50, 100, 400]:
        X = np.concatenate([np.ones((R, n, 1)), rng.standard_normal((R, n, 2))], axis=2)
        sd = np.sqrt(0.1 + X[..., 1] ** 2 + X[..., 2] ** 2)
        y = 2 + sd * rng.standard_normal((R, n))
        fit = fit_batch(X, y)
        Rm, r = np.array([[0, 1, 0], [0, 0, 1]]), np.zeros(2)
        row = {"n": n}
        for c in ["classical", "HC0", "HC3"]:
            V = covariance(X, fit.resid, c, bread=fit.bread, lev=fit.lev)
            _, p = wald_test(fit.beta, V, Rm, r, df_resid=n - 3)
            row[f"rejection_{c}"] = np.mean(p < 0.05)
        rows.append(row)
    return pd.DataFrame(rows)


def main(R=5000):
    setup()
    print("Experiment 4: robust inference")
    hc_comparison(R).to_csv(TAB_DIR / "04a_hc_comparison.csv", index=False)
    wls_efficiency(R).to_csv(TAB_DIR / "04b_wls_efficiency.csv", index=False)
    hac_vs_persistence(R).to_csv(TAB_DIR / "04c_hac_persistence.csv", index=False)
    joint_test_size(R).to_csv(TAB_DIR / "04d_joint_test_size.csv", index=False)


if __name__ == "__main__":
    main()
