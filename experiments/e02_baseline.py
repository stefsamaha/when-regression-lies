"""
Experiment 2: when every assumption holds, does OLS deliver what theory promises?

True model: y = 2 + 3x + eps,  x ~ N(0,1),  eps ~ N(0,1).

Checks:
  2a. Unbiasedness and the exact variance:  E[b1] = 3  and  Var(b1) = sigma^2 / (n - 3)
      (with Gaussian x, sum (x - xbar)^2 ~ chi2(n-1), and E[1/chi2_k] = 1/(k-2)).
  2b. The t-statistic follows Student t(n-2) exactly, and using the normal
      critical value 1.96 at small n gives under-coverage.
  2c. Under H0 (true slope 0) p-values are Uniform(0,1): exactly 5% fall below 0.05.
      Same for the F-test of a joint hypothesis.
  2d. Gauss-Markov: OLS beats another linear unbiased estimator.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from regression_lies import OLS, dgp, montecarlo, fit_batch, wald_test
from regression_lies.ols import covariance
from regression_lies.plotting import setup, save, TAB_DIR, BLUE, ORANGE, AQUA, MUTED, INK2

NS = [5, 10, 20, 50, 100, 500, 1000]


def consistency_table(R, seed=100):
    rows = []
    for i, n in enumerate(NS):
        res = montecarlo.run(dgp.baseline, n=n, R=R, seed=seed + i, keep_draws=True)
        t = res["t_classical"]
        rows.append({
            "n": n,
            "mean_b1": res["mean"],
            "bias": res["bias"],
            "emp_sd": res["emp_sd"],
            "theory_sd": np.sqrt(1 / (n - 3)) if n > 3 else np.nan,
            "mean_se": res["mean_se_classical"],
            "coverage_t": res["coverage_classical"],
            "coverage_z": np.mean(np.abs(t) <= 1.96),
        })
    return pd.DataFrame(rows)


def plot_sampling(R, seed=200):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    # (a) sampling distribution of b1 at several n
    ax = axes[0]
    for n, col in [(10, ORANGE), (50, AQUA), (500, BLUE)]:
        res = montecarlo.run(dgp.baseline, n=n, R=R, seed=seed + n, keep_draws=True)
        ax.hist(res["draws"], bins=80, density=True, histtype="step", color=col, lw=1.6, label=f"n = {n}")
    ax.axvline(3, color=INK2, lw=1, ls="--")
    ax.set_xlim(1.8, 4.2)
    ax.set_xlabel(r"$\hat\beta_1$ over 5,000 samples")
    ax.set_title("(a) Unbiased and consistent:\ncentred on 3, shrinking like $1/\\sqrt{n}$")
    ax.legend()

    # (b) t statistic at n = 5 vs t(3) and N(0,1)
    ax = axes[1]
    res = montecarlo.run(dgp.baseline, n=5, R=R * 4, seed=seed, keep_draws=True)
    t = res["t_classical"]
    grid = np.linspace(-6, 6, 400)
    ax.hist(t, bins=np.linspace(-6, 6, 90), density=True, color=BLUE, alpha=0.35, label="simulated t, n = 5")
    ax.plot(grid, stats.t.pdf(grid, 3), color=BLUE, label="Student t(3)")
    ax.plot(grid, stats.norm.pdf(grid), color=ORANGE, ls="--", label="N(0, 1)")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 0.6)
    ax.set_xlabel(r"$t = (\hat\beta_1 - 3)/\widehat{se}$")
    ax.set_title("(b) Why Student t: estimating $\\sigma$\nfattens the tails at small n")
    ax.legend(fontsize=8, loc="lower center")

    # (c) p-values under H0 are uniform
    ax = axes[2]
    rng = np.random.default_rng(seed + 1)
    X, y, _ = dgp.baseline(rng, R, 30, beta=np.array([2.0, 0.0]))
    fit = fit_batch(X, y)
    se = np.sqrt(fit.sigma2 * fit.bread[:, 1, 1])
    p = 2 * stats.t.sf(np.abs(fit.beta[:, 1] / se), 28)
    ax.hist(p, bins=20, range=(0, 1), color=BLUE, alpha=0.8, rwidth=0.92)
    ax.axhline(R / 20, color=INK2, ls="--", lw=1)
    ax.text(0.02, R / 20 * 1.22, "Uniform(0,1) expectation", color=INK2, fontsize=8)
    ax.set_xlabel("p-value for H0: slope = 0 (true)")
    ax.set_ylabel("count")
    ax.set_title(f"(c) Under a true null, p < 0.05\n{np.mean(p < 0.05):.1%} of the time (target 5%)")
    fig.tight_layout()
    save(fig, "02_baseline_sampling.png")


def f_test_size(R, seed=300):
    """Joint test H0: b1 = b2 = 0 in y = 2 + 0 x1 + 0 x2 + eps, n = 25."""
    rng = np.random.default_rng(seed)
    n = 25
    X = np.concatenate([np.ones((R, n, 1)), rng.standard_normal((R, n, 2))], axis=2)
    y = 2 + rng.standard_normal((R, n))
    fit = fit_batch(X, y)
    V = covariance(X, fit.resid, "classical", bread=fit.bread)
    Rm = np.array([[0, 1, 0], [0, 0, 1]])
    _, p = wald_test(fit.beta, V, Rm, np.zeros(2), df_resid=n - 3)
    return np.mean(p < 0.05)


def gauss_markov(R, seed=400, n=40):
    """Compare OLS with the 'grouping' estimator (Wald, 1940):
    b_G = (mean y in top half of x - mean y in bottom half) / (same for x).
    It is linear in y and unbiased given X, so Gauss-Markov says Var(b_G) >= Var(b_OLS).
    """
    rng = np.random.default_rng(seed)
    X, y, _ = dgp.baseline(rng, R, n)
    b_ols = fit_batch(X, y).beta[:, 1]
    x = X[..., 1]
    order = np.argsort(x, axis=1)
    xs, ys = np.take_along_axis(x, order, 1), np.take_along_axis(y, order, 1)
    h = n // 2
    b_g = (ys[:, h:].mean(1) - ys[:, :h].mean(1)) / (xs[:, h:].mean(1) - xs[:, :h].mean(1))
    return pd.DataFrame({
        "estimator": ["OLS", "Grouping (Wald 1940)"],
        "mean": [b_ols.mean(), b_g.mean()],
        "sd": [b_ols.std(ddof=1), b_g.std(ddof=1)],
    })


def main(R=5000):
    setup()
    print("Experiment 2: baseline (all assumptions hold)")
    tab = consistency_table(R)
    tab.to_csv(TAB_DIR / "02a_baseline_consistency.csv", index=False)
    plot_sampling(R)
    extra = pd.Series({"F_test_size_n25": f_test_size(R)})
    extra.to_csv(TAB_DIR / "02c_f_test_size.csv")
    gauss_markov(R).to_csv(TAB_DIR / "02d_gauss_markov.csv", index=False)


if __name__ == "__main__":
    main()
