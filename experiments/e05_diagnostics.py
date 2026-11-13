"""
Experiment 5: can we *see* a broken assumption in one dataset?

5a. Residual-vs-fitted and normal Q-Q plots for four single datasets:
    valid, heteroskedastic, non-linear truth (A1 broken), skewed errors (A5 broken).
5b. Influence: one high-leverage point that drags the slope. Leverage h_ii,
    studentised residuals and Cook's distance identify it.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from regression_lies import OLS
from regression_lies.plotting import setup, save, TAB_DIR, BLUE, ORANGE, AQUA, INK2, MUTED


def _datasets(n=200, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    return {
        "Valid": (x, 2 + 3 * x + z),
        "Heteroskedastic": (x, 2 + 3 * x + np.sqrt(0.1 + x**2) * z),
        "Non-linear truth\n(fit is linear)": (x, 2 + 3 * x + 1.5 * x**2 + 0.6 * z),
        "Skewed errors": (x, 2 + 3 * x + (rng.chisquare(1, n) - 1) / np.sqrt(2)),
    }


def residual_panels():
    data = _datasets()
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
    for j, (name, (x, y)) in enumerate(data.items()):
        X = np.column_stack([np.ones_like(x), x])
        m = OLS().fit(X, y)
        r = m.studentized_residuals()

        ax = axes[0, j]
        ax.scatter(m.fitted, r, s=12, color=BLUE, alpha=0.6, edgecolor="none")
        ax.axhline(0, color=INK2, lw=1)
        # running mean shows curvature; running spread shows heteroskedasticity
        order = np.argsort(m.fitted)
        k = 25
        fs = m.fitted[order]
        rm = np.convolve(r[order], np.ones(k) / k, mode="valid")
        rs = np.sqrt(np.convolve(r[order] ** 2, np.ones(k) / k, mode="valid"))
        mid = fs[k // 2: k // 2 + len(rm)]
        ax.plot(mid, rm, color=ORANGE, lw=2, label="running mean")
        ax.plot(mid, rs, color=AQUA, lw=1.5, ls="--", label="running RMS")
        ax.plot(mid, -rs, color=AQUA, lw=1.5, ls="--")
        ax.set_title(name)
        ax.set_xlabel("fitted value")
        if j == 0:
            ax.set_ylabel("studentised residual")
            ax.legend(fontsize=8, loc="upper left")

        ax = axes[1, j]
        (osm, osr), (slope, icpt, _) = stats.probplot(r, dist="norm")
        ax.scatter(osm, osr, s=12, color=BLUE, alpha=0.6, edgecolor="none")
        ax.plot(osm, icpt + slope * osm, color=INK2, lw=1)
        ax.set_xlabel("normal quantiles")
        if j == 0:
            ax.set_ylabel("residual quantiles (Q-Q)")
    fig.suptitle("Diagnostics on a single dataset (n = 200): what each violation looks like",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, "05a_residual_diagnostics.png")


def influence(seed=8, n=30):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 3, n)
    y = 2 + 3 * x + rng.standard_normal(n)
    x = np.append(x, 8.0)           # far from the other x values: high leverage
    y = np.append(y, 10.0)          # and far below the true line (true value would be 26)
    X = np.column_stack([np.ones_like(x), x])
    m_all = OLS().fit(X, y)
    m_drop = OLS().fit(X[:-1], y[:-1])
    h, r, D = m_all.leverage, m_all.studentized_residuals(), m_all.cooks_distance()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    ax.scatter(x[:-1], y[:-1], s=22, color=BLUE, label="regular points")
    ax.scatter(x[-1:], y[-1:], s=60, color=ORANGE, marker="D", label="high-leverage point")
    grid = np.linspace(0, 8.5, 50)
    ax.plot(grid, 2 + 3 * grid, color=INK2, ls="--", lw=1, label="truth: 2 + 3x")
    ax.plot(grid, m_all.beta[0] + m_all.beta[1] * grid, color=ORANGE,
            label=f"OLS with it: slope {m_all.beta[1]:.2f}")
    ax.plot(grid, m_drop.beta[0] + m_drop.beta[1] * grid, color=BLUE,
            label=f"OLS without it: slope {m_drop.beta[1]:.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("(a) One point out of 31 rotates the line")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    sizes = 30 + 600 * D / D.max()
    ax.scatter(h[:-1], r[:-1], s=sizes[:-1], color=BLUE, alpha=0.6, edgecolor="white")
    ax.scatter(h[-1:], r[-1:], s=sizes[-1:], color=ORANGE, alpha=0.8, marker="D", edgecolor="white")
    ax.axvline(2 * 2 / len(x), color=MUTED, ls=":", lw=1)
    ax.text(2 * 2 / len(x) + 0.01, ax.get_ylim()[0] * 0.9 if r.min() < 0 else 0.1,
            "2p/n rule", color=INK2, fontsize=8)
    ax.annotate(f"Cook's D = {D[-1]:.1f}\n(others < {np.sort(D)[-2]:.2f})", (h[-1], r[-1]),
                textcoords="offset points", xytext=(-110, 10), fontsize=8.5, color=INK2)
    ax.set_xlabel(r"leverage $h_{ii}$")
    ax.set_ylabel("studentised residual")
    ax.set_title("(b) Influence = leverage x surprise\n(bubble area proportional to Cook's distance)")
    fig.tight_layout()
    save(fig, "05b_influence.png")

    return pd.DataFrame({
        "quantity": ["slope with point", "slope without point", "leverage of point",
                     "mean leverage (p/n)", "Cook's D of point", "max Cook's D of others"],
        "value": [m_all.beta[1], m_drop.beta[1], h[-1], 2 / len(x), D[-1], np.sort(D)[-2]],
    })


def main(R=None):
    setup()
    print("Experiment 5: diagnostics")
    residual_panels()
    influence().to_csv(TAB_DIR / "05_influence.csv", index=False)


if __name__ == "__main__":
    main()
