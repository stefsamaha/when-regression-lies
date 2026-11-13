"""
Experiment 3: break one assumption at a time and measure the damage.

For every scenario and sample size we record, for the slope:
    bias, empirical sd, the classical SE's honesty ratio (mean se / empirical sd),
    95% CI coverage with classical, HC3 and Newey-West (HAC) standard errors,
    and the false-positive rate of a t-test of "slope = 0" when the true slope is 0.

The point of the experiment is that the failures come in two very different kinds:

    A3 (endogeneity)      the *estimate* is wrong. No standard error can fix it,
                          and more data makes the false confidence worse.
    A4 (non-spherical)    the estimate is fine, the *standard error* is wrong.
                          The right covariance estimator fixes it.
    A5 (non-normality)    nothing is wrong asymptotically; only small-n
                          inference is approximate.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from regression_lies import dgp, montecarlo, fit_batch, standard_errors
from regression_lies.plotting import (setup, save, TAB_DIR, BLUE, ORANGE, AQUA, YELLOW,
                                      MAGENTA, INK2, MUTED, GRID)

NS = [10, 25, 100, 1000]
COVS = ("classical", "HC3", "HAC")


def false_positive_rate(dgp_fn, n, R, seed, cov_type="classical"):
    """Share of samples in which we 'discover' a slope when the true slope is 0."""
    rng = np.random.default_rng(seed)
    X, y, _ = dgp_fn(rng, R, n, beta=np.array([2.0, 0.0]))
    fit = fit_batch(X, y)
    se = standard_errors(X, fit, cov_type)[:, 1]
    p = 2 * stats.t.sf(np.abs(fit.beta[:, 1] / se), n - 2)
    return np.mean(p < 0.05)


def run_grid(R):
    rows = []
    for si, (name, fn) in enumerate(dgp.SCENARIOS.items()):
        for ni, n in enumerate(NS):
            seed = 1000 * (si + 1) + ni
            res = montecarlo.run(fn, n=n, R=R, seed=seed, cov_types=COVS)
            row = {
                "scenario": name,
                "violates": dgp.VIOLATES[name],
                "n": n,
                "mean_b1": res["mean"],
                "bias": res["bias"],
                "emp_sd": res["emp_sd"],
                "se_ratio_classical": res["se_ratio_classical"],
            }
            for c in COVS:
                row[f"coverage_{c}"] = res[f"coverage_{c}"]
            row["false_pos_classical"] = false_positive_rate(fn, n, R, seed + 500)
            row["false_pos_HC3"] = false_positive_rate(fn, n, R, seed + 500, "HC3")
            row["false_pos_HAC"] = false_positive_rate(fn, n, R, seed + 500, "HAC")
            rows.append(row)
            print(f"  {name:<22} n={n:<5} bias={row['bias']:+.3f} "
                  f"cov(classical)={row['coverage_classical']:.3f}")
    return pd.DataFrame(rows)


def plot_coverage_dots(df, n=100):
    """Headline figure: coverage of nominal 95% intervals, by scenario and SE type."""
    d = df[df.n == n].set_index("scenario").loc[list(dgp.SCENARIOS)]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ypos = np.arange(len(d))[::-1]
    ax.axvspan(0.94, 0.96, color=GRID, zorder=0)
    ax.axvline(0.95, color=INK2, lw=1, ls="--", zorder=1)
    for c, col, lab, off in [("classical", ORANGE, "Classical SE", 0.18),
                             ("HC3", BLUE, "HC3 robust SE", 0.0),
                             ("HAC", AQUA, "Newey-West HAC SE", -0.18)]:
        ax.plot(d[f"coverage_{c}"], ypos + off, "o", color=col, ms=8, label=lab,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{s}\n({d.loc[s, 'violates']})" for s in d.index], fontsize=8.5)
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel(f"share of 95% confidence intervals containing the true slope (n = {n}, 5,000 samples)")
    ax.set_title("Coverage of nominal 95% intervals when one assumption breaks")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "03a_coverage_by_scenario.png")


def plot_endogeneity(df, R, seed=77):
    """Endogeneity: the estimator converges to the wrong number and CIs close in on it."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    endo = [("Omitted variable", ORANGE), ("Measurement error", BLUE), ("Simultaneity", AQUA)]
    for name, col in endo:
        res = montecarlo.run(dgp.SCENARIOS[name], n=500, R=R, seed=seed, keep_draws=True)
        ax.hist(res["draws"], bins=70, density=True, histtype="stepfilled", alpha=0.35, color=col)
        ax.hist(res["draws"], bins=70, density=True, histtype="step", color=col, lw=1.4, label=name)
    res = montecarlo.run(dgp.baseline, n=500, R=R, seed=seed, keep_draws=True)
    ax.hist(res["draws"], bins=70, density=True, histtype="step", color=INK2, lw=1.4, label="Valid (reference)")
    ax.axvline(3, color=INK2, ls="--", lw=1)
    ax.text(3.02, ax.get_ylim()[1] * 0.92, "true slope = 3", color=INK2, fontsize=8)
    ax.set_xlabel(r"$\hat\beta_1$  (n = 500)")
    ax.set_title("(a) Endogeneity moves the whole distribution")
    ax.legend(fontsize=8)

    ax = axes[1]
    styles = {"Omitted variable": ("s--", 9), "Measurement error": ("o-", 6),
              "Simultaneity": ("o-", 6), "Valid assumptions": ("o-", 6)}
    for name, col in endo + [("Valid assumptions", INK2)]:
        d = df[df.scenario == name]
        mk, ms = styles[name]
        ax.plot(d.n, d.coverage_HC3, mk, color=col, label=name, ms=ms, zorder=2 if "Omitted" in name else 3)
    ax.text(0.98, 0.55, "omitted variable and simultaneity\nhave the same bias-to-noise ratio here,\nso their curves coincide",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
    ax.axhline(0.95, color=INK2, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("sample size n")
    ax.set_ylabel("95% CI coverage (HC3 robust SE)")
    ax.set_title("(b) More data makes it worse: intervals shrink\naround the wrong value")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "03b_endogeneity.png")


def plot_false_positives(df):
    """Spurious significance: rejecting a true null of zero effect."""
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for name, col in [("Autocorrelated", ORANGE), ("Heteroskedastic", BLUE),
                      ("Heavy-tailed errors", AQUA), ("Valid assumptions", INK2)]:
        d = df[df.scenario == name]
        ax.plot(d.n, d.false_pos_classical, "o-", color=col, label=f"{name}, classical SE")
        if name in ("Autocorrelated", "Heteroskedastic"):
            fix = "false_pos_HAC" if name == "Autocorrelated" else "false_pos_HC3"
            lab = "HAC" if name == "Autocorrelated" else "HC3"
            ax.plot(d.n, d[fix], "s:", color=col, ms=6, label=f"{name}, {lab} SE")
    ax.axhline(0.05, color=INK2, ls="--", lw=1)
    ax.text(10.5, 0.062, "nominal 5%", color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("false-positive rate (p < 0.05, true slope = 0)")
    ax.set_title("Spurious significance: how often a zero effect looks real")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    save(fig, "03c_false_positives.png")


def cauchy_table(R, seed=909):
    """Infinite-variance errors: no amount of data makes OLS concentrate."""
    def cauchy(rng, R_, n, beta=dgp.BETA):
        return dgp.non_gaussian(rng, R_, n, beta, "cauchy")
    rows = []
    for i, n in enumerate(NS):
        res = montecarlo.run(cauchy, n=n, R=R, seed=seed + i, keep_draws=True, cov_types=("classical", "HC3"))
        b = res["draws"]
        q25, q50, q75 = np.percentile(b, [25, 50, 75])
        rows.append({"n": n, "median_b1": q50, "IQR_b1": q75 - q25,
                     "share_off_by_more_than_1": np.mean(np.abs(b - 3) > 1),
                     "coverage_classical": res["coverage_classical"], "coverage_HC3": res["coverage_HC3"]})
    return pd.DataFrame(rows)


def main(R=5000):
    setup()
    print("Experiment 3: assumption violations")
    cauchy_table(R).to_csv(TAB_DIR / "03d_cauchy.csv", index=False)
    df = run_grid(R)
    df.to_csv(TAB_DIR / "03_violations.csv", index=False)
    plot_coverage_dots(df)
    plot_endogeneity(df, R)
    plot_false_positives(df)
    return df


if __name__ == "__main__":
    main()
