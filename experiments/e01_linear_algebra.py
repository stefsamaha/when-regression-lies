"""
Experiment 1: the linear algebra of least squares.

1a. Projection geometry: y_hat is the orthogonal projection of y onto col(X).
1b. Numerical accuracy: normal equations vs QR vs SVD vs LAPACK lstsq as the
    design becomes ill-conditioned.
1c. Multicollinearity: an exact Monte Carlo check of the variance inflation factor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regression_lies import OLS, hat_matrix, dgp, montecarlo
from regression_lies.ols import SOLVERS
from regression_lies.plotting import setup, save, TAB_DIR, BLUE, ORANGE, AQUA, YELLOW, INK2, MUTED


def projection_geometry():
    """n = 3 observations, p = 2 columns: col(X) is a plane in R^3 we can draw."""
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([0.3, 3.6, 1.2])
    m = OLS().fit(X, y)
    H = hat_matrix(X)

    checks = {
        "||H - H'||": np.abs(H - H.T).max(),
        "||H^2 - H||": np.abs(H @ H - H).max(),
        "trace(H) - p": np.trace(H) - X.shape[1],
        "max |X'e|": np.abs(X.T @ m.resid).max(),
        "y_hat . e": m.fitted @ m.resid,
        "||y||^2 - ||y_hat||^2 - ||e||^2": y @ y - m.fitted @ m.fitted - m.resid @ m.resid,
    }

    # Rotate R^3 with the full QR basis so col(X) becomes the horizontal plane z = 0.
    # Q is orthogonal, so lengths and angles are preserved: the picture is exact.
    Qfull, _ = np.linalg.qr(X, mode="complete")
    if (Qfull[:, 2] @ m.resid) < 0:          # orient the normal direction so e points "up"
        Qfull[:, 2] *= -1
    rot = lambda v: Qfull.T @ v
    Y, Yh, E = rot(y), rot(m.fitted), rot(m.resid)
    X1, X2 = rot(X[:, 0]), rot(X[:, 1])

    fig = plt.figure(figsize=(7, 5.6))
    ax = fig.add_subplot(projection="3d")
    pts = np.array([Y, Yh, X1, X2, np.zeros(3)])
    lo, hi = pts[:, :2].min() - 0.8, pts[:, :2].max() + 0.8
    lim = max(abs(lo), abs(hi))
    gx, gy = np.meshgrid(np.linspace(-lim, lim, 2), np.linspace(-lim, lim, 2))
    ax.plot_surface(gx, gy, 0 * gx, color=BLUE, alpha=0.10, linewidth=0)
    ax.text(-lim, lim * 0.85, 0, "column space of X", color=BLUE, fontsize=9)
    o = np.zeros(3)

    def arrow(v, c, lab, start=o, off=(0.15, 0.15, 0.15)):
        ax.quiver(*start, *v, color=c, arrow_length_ratio=0.07, linewidth=2.2)
        ax.text(*(start + v + np.array(off)), lab, color=c, fontsize=11, weight="bold")

    arrow(X1, MUTED, r"$x_1$", off=(0.1, 0.35, 0.1))
    arrow(X2, MUTED, r"$x_2$", off=(0.2, -0.5, 0.0))
    arrow(Y, ORANGE, r"$y$")
    arrow(Yh, BLUE, r"$\hat y = Hy$", off=(0.1, 0.2, -0.45))
    arrow(E, AQUA, r"$e=(I-H)y$", start=Yh, off=(0.15, 0.1, -0.6))
    # right-angle marker where e meets the plane
    d = 0.25 * Yh / np.linalg.norm(Yh)
    up = 0.25 * E / np.linalg.norm(E)
    corner = np.array([Yh - d, Yh - d + up, Yh + up])
    ax.plot(corner[:, 0], corner[:, 1], corner[:, 2], color=INK2, lw=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, Y[2] * 1.3)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.view_init(elev=20, azim=-60)
    ax.set_title("Least squares is an orthogonal projection\n"
                 "the residual meets the column space of X at a right angle")
    save(fig, "01a_projection_geometry.png")
    return pd.Series(checks, name="value")


def solver_accuracy():
    """Polynomial design on [0, 1]: cond(X) explodes with the degree.

    y = X beta exactly (no noise), so any error is pure floating-point error.
    Theory: QR/SVD error ~ eps * cond(X);  normal equations ~ eps * cond(X)^2.
    """
    x = np.linspace(0, 1, 100)
    rows = []
    for deg in range(1, 16):
        X = np.vander(x, deg + 1, increasing=True)
        beta = np.ones(deg + 1)
        y = X @ beta
        row = {"degree": deg, "cond": np.linalg.cond(X)}
        for method in SOLVERS:
            try:
                b = OLS(method).fit(X, y).beta
                row[method] = np.linalg.norm(b - beta) / np.linalg.norm(beta)
            except np.linalg.LinAlgError:
                row[method] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    eps = np.finfo(float).eps
    c = df["cond"]
    ax.loglog(c, eps * c, color=MUTED, ls=":", lw=1.3)
    ax.loglog(c, eps * c**2, color=MUTED, ls="--", lw=1.3)
    ax.text(c.iloc[-3], eps * c.iloc[-3] * 3, r"$\epsilon\,\kappa$", color=INK2)
    ax.text(c.iloc[6], eps * c.iloc[6] ** 2 * 3, r"$\epsilon\,\kappa^2$", color=INK2)
    for method, col, lab in [("normal", ORANGE, "Normal equations"), ("qr", BLUE, "QR"),
                             ("svd", AQUA, "SVD"), ("lstsq", YELLOW, "LAPACK lstsq")]:
        ax.loglog(c, df[method].clip(lower=1e-17), "o-", color=col, label=lab, ms=4, lw=1.6)
    ax.set_ylim(1e-17, 1e3)
    ax.set_xlabel(r"condition number $\kappa(X)$  (polynomial degree 1 to 15)")
    ax.set_ylabel(r"relative error  $\|\hat\beta-\beta\| / \|\beta\|$")
    ax.set_title("Normal equations square the condition number")
    ax.legend(loc="upper left")
    save(fig, "01b_solver_accuracy.png")
    return df


def multicollinearity(R, seed=11):
    rows = []
    for i, rho in enumerate([0.0, 0.5, 0.8, 0.9, 0.95, 0.99]):
        res = montecarlo.run(dgp.collinear, n=100, R=R, seed=seed + i, coef=1, rho=rho)
        # Var(b1) for x ~ N with corr rho, homoskedastic sigma=1:  1 / ((n - p - 1) (1 - rho^2))
        theory_sd = np.sqrt(1 / ((100 - 3 - 1) * (1 - rho**2)))
        rows.append({"rho": rho, "VIF": 1 / (1 - rho**2), "mean_b1": res["mean"], "bias": res["bias"],
                     "emp_sd": res["emp_sd"], "theory_sd": theory_sd,
                     "mean_se": res["mean_se_classical"], "coverage": res["coverage_classical"]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(df["rho"], df["theory_sd"], color=MUTED, ls="--", label=r"theory  $\sigma\sqrt{\mathrm{VIF}/(n-p-1)}$")
    ax.plot(df["rho"], df["emp_sd"], "o", color=BLUE, ms=8, label="Monte Carlo sd of slope")
    for _, r in df.iterrows():
        ax.annotate(f"cov {r.coverage:.3f}", (r.rho, r.emp_sd), textcoords="offset points",
                    xytext=(-62, 4), color=INK2, fontsize=8)
    ax.set_xlabel(r"correlation between regressors  $\rho$")
    ax.set_ylabel("standard deviation of slope estimate")
    ax.set_title("Multicollinearity inflates variance, not bias\n(CI coverage stays at 95%)")
    ax.legend(loc="upper left")
    save(fig, "01c_multicollinearity.png")
    return df


def main(R=5000):
    setup()
    print("Experiment 1: linear algebra")
    projection_geometry().to_csv(TAB_DIR / "01a_projection_identities.csv")
    solver_accuracy().to_csv(TAB_DIR / "01b_solver_accuracy.csv", index=False)
    multicollinearity(R).to_csv(TAB_DIR / "01c_multicollinearity.csv", index=False)


if __name__ == "__main__":
    main()
