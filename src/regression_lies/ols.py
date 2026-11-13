"""
Ordinary least squares from scratch.

Everything here is built on NumPy linear algebra only. statsmodels is used
exclusively in the test suite, as an independent reference.

All inference functions work on *stacked* problems: X has shape (R, n, p) and
y has shape (R, n), so thousands of Monte Carlo regressions can be fitted in
one vectorised call. A single regression is just the case R = 1, which is what
the `OLS` class wraps.

Notation
--------
n : number of observations
p : number of regressors (including the intercept column)
beta_hat = argmin ||y - X b||^2
H = X (X'X)^{-1} X'           the hat (projection) matrix
e = (I - H) y                 residuals
s^2 = e'e / (n - p)           unbiased estimate of the error variance
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

SOLVERS = ("qr", "normal", "lstsq", "svd")
COV_TYPES = ("classical", "HC0", "HC1", "HC2", "HC3", "HAC")


# ---------------------------------------------------------------------------
# 1. Solving the least-squares problem
# ---------------------------------------------------------------------------

def solve_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """beta = (X'X)^{-1} X'y, solved via a linear system (never an explicit inverse).

    Fast and simple, but it squares the condition number: cond(X'X) = cond(X)^2.
    """
    XtX = np.swapaxes(X, -1, -2) @ X
    Xty = np.swapaxes(X, -1, -2) @ y[..., None]
    return np.linalg.solve(XtX, Xty)[..., 0]


def solve_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Thin QR: X = QR, so ||y - Xb|| is minimised by R b = Q'y.

    Q has orthonormal columns, so it does not distort lengths; the problem keeps
    cond(R) = cond(X) instead of cond(X)^2.
    """
    Q, R = np.linalg.qr(X)
    Qty = np.swapaxes(Q, -1, -2) @ y[..., None]
    return np.linalg.solve(R, Qty)[..., 0]


def solve_svd(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """X = U S V', beta = V S^{-1} U'y (the Moore-Penrose pseudo-inverse solution)."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    Uty = np.swapaxes(U, -1, -2) @ y[..., None]
    return (np.swapaxes(Vt, -1, -2) @ (Uty / s[..., None]))[..., 0]


def solve_lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LAPACK's least-squares driver (gelsd, SVD based). 2-D input only."""
    if X.ndim == 3:
        return np.stack([np.linalg.lstsq(Xi, yi, rcond=None)[0] for Xi, yi in zip(X, y)])
    return np.linalg.lstsq(X, y, rcond=None)[0]


_SOLVER_FUNCS = {
    "qr": solve_qr,
    "normal": solve_normal_equations,
    "lstsq": solve_lstsq,
    "svd": solve_svd,
}


def solve(X: np.ndarray, y: np.ndarray, method: str = "qr") -> np.ndarray:
    if method not in _SOLVER_FUNCS:
        raise ValueError(f"method must be one of {SOLVERS}")
    return _SOLVER_FUNCS[method](X, y)


# ---------------------------------------------------------------------------
# 2. Geometry: the hat matrix
# ---------------------------------------------------------------------------

def hat_matrix(X: np.ndarray) -> np.ndarray:
    """H = Q Q' where X = QR. Orthogonal projector onto col(X)."""
    Q, _ = np.linalg.qr(X)
    return Q @ np.swapaxes(Q, -1, -2)


def leverage(X: np.ndarray) -> np.ndarray:
    """Diagonal of H without forming the n x n matrix: h_ii = ||row i of Q||^2."""
    Q, _ = np.linalg.qr(X)
    return np.sum(Q**2, axis=-1)


# ---------------------------------------------------------------------------
# 3. Covariance estimators for beta_hat
# ---------------------------------------------------------------------------

def _bread(X: np.ndarray) -> np.ndarray:
    """(X'X)^{-1} computed from R: (X'X)^{-1} = R^{-1} R^{-T}."""
    _, R = np.linalg.qr(X)
    Rinv = np.linalg.inv(R)
    return Rinv @ np.swapaxes(Rinv, -1, -2)


def newey_west_lags(n: int) -> int:
    """Rule-of-thumb bandwidth, floor(4 (n/100)^(2/9)) (Newey and West, 1994)."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def covariance(
    X: np.ndarray,
    resid: np.ndarray,
    cov_type: str = "classical",
    hac_lags: int | None = None,
    bread: np.ndarray | None = None,
    lev: np.ndarray | None = None,
) -> np.ndarray:
    """Estimated Var(beta_hat | X), stacked over the leading axis.

    classical : s^2 (X'X)^{-1}                   valid under homoskedastic, uncorrelated errors
    HC0       : (X'X)^{-1} X' diag(e_i^2) X (X'X)^{-1}           White (1980)
    HC1       : HC0 * n / (n - p)                                 small-sample scaling
    HC2       : e_i^2 replaced by e_i^2 / (1 - h_ii)
    HC3       : e_i^2 replaced by e_i^2 / (1 - h_ii)^2            close to the jackknife
    HAC       : Newey-West with Bartlett weights, robust to heteroskedasticity
                and autocorrelation up to `hac_lags`
    """
    squeeze = X.ndim == 2
    if squeeze:
        X, resid = X[None], resid[None]
        bread = None if bread is None else bread[None]
        lev = None if lev is None else lev[None]

    R, n, p = X.shape
    B = _bread(X) if bread is None else bread

    if cov_type == "classical":
        s2 = np.sum(resid**2, axis=1) / (n - p)
        V = s2[:, None, None] * B

    elif cov_type in ("HC0", "HC1", "HC2", "HC3"):
        w = resid**2
        if cov_type in ("HC2", "HC3"):
            h = leverage(X) if lev is None else lev
            w = w / (1 - h) if cov_type == "HC2" else w / (1 - h) ** 2
        meat = np.einsum("rni,rn,rnj->rij", X, w, X)
        V = B @ meat @ B
        if cov_type == "HC1":
            V = V * n / (n - p)

    elif cov_type == "HAC":
        L = newey_west_lags(n) if hac_lags is None else hac_lags
        u = X * resid[..., None]                      # score contributions x_t e_t
        meat = np.einsum("rti,rtj->rij", u, u)        # lag 0
        for lag in range(1, L + 1):
            weight = 1 - lag / (L + 1)                # Bartlett kernel keeps the matrix PSD
            G = np.einsum("rti,rtj->rij", u[:, lag:], u[:, :-lag])
            meat += weight * (G + np.swapaxes(G, -1, -2))
        V = B @ meat @ B

    else:
        raise ValueError(f"cov_type must be one of {COV_TYPES}")

    return V[0] if squeeze else V


# ---------------------------------------------------------------------------
# 4. Fitting many regressions at once
# ---------------------------------------------------------------------------

@dataclass
class BatchFit:
    beta: np.ndarray        # (R, p)
    resid: np.ndarray       # (R, n)
    bread: np.ndarray       # (R, p, p)  (X'X)^{-1}
    lev: np.ndarray         # (R, n)
    n: int
    p: int

    @property
    def df_resid(self) -> int:
        return self.n - self.p

    @property
    def sigma2(self) -> np.ndarray:
        return np.sum(self.resid**2, axis=1) / self.df_resid


def fit_batch(X: np.ndarray, y: np.ndarray) -> BatchFit:
    """QR-based OLS for a stack of R independent datasets."""
    if X.ndim == 2:
        X, y = X[None], y[None]
    Q, Rm = np.linalg.qr(X)
    Qty = np.swapaxes(Q, -1, -2) @ y[..., None]
    beta = np.linalg.solve(Rm, Qty)[..., 0]
    fitted = (Q @ Qty)[..., 0]                         # H y = Q Q'y
    Rinv = np.linalg.inv(Rm)
    bread = Rinv @ np.swapaxes(Rinv, -1, -2)
    lev = np.sum(Q**2, axis=-1)
    R, n, p = X.shape
    return BatchFit(beta=beta, resid=y - fitted, bread=bread, lev=lev, n=n, p=p)


def standard_errors(X, fit: BatchFit, cov_type="classical", hac_lags=None) -> np.ndarray:
    V = covariance(X, fit.resid, cov_type, hac_lags, bread=fit.bread, lev=fit.lev)
    return np.sqrt(np.diagonal(V, axis1=-2, axis2=-1))


# ---------------------------------------------------------------------------
# 5. Hypothesis tests
# ---------------------------------------------------------------------------

def wald_test(beta, V, Rmat, r, df_resid=None):
    """Test H0: R beta = r with q restrictions.

    W = (R b - r)' [R V R']^{-1} (R b - r)
    F = W / q.  With the classical V and Gaussian errors, F ~ F(q, n - p) exactly.
    With a robust V the F reference is an approximation that improves with n.

    Returns (F statistic, p-value), stacked if beta is stacked.
    """
    Rmat = np.atleast_2d(Rmat)
    q = Rmat.shape[0]
    diff = beta @ Rmat.T - r                          # (..., q)
    RVR = Rmat @ V @ Rmat.T                           # (..., q, q)
    W = np.einsum("...i,...i->...", diff, np.linalg.solve(RVR, diff[..., None])[..., 0])
    F = W / q
    pval = stats.f.sf(F, q, df_resid) if df_resid is not None else stats.chi2.sf(W, q)
    return F, pval


# ---------------------------------------------------------------------------
# 6. A friendly single-dataset interface
# ---------------------------------------------------------------------------

@dataclass
class OLS:
    """OLS for one dataset.

    >>> model = OLS(method="qr").fit(X, y)
    >>> print(model.summary(cov_type="HC3"))
    """

    method: str = "qr"
    beta: np.ndarray = field(init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OLS":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.X, self.y = X, y
        self.n, self.p = X.shape
        self.beta = solve(X, y, self.method)
        self.fitted = X @ self.beta
        self.resid = y - self.fitted
        self.df_resid = self.n - self.p
        self.sigma2 = self.resid @ self.resid / self.df_resid
        return self

    # geometry ---------------------------------------------------------------
    @property
    def leverage(self) -> np.ndarray:
        return leverage(self.X)

    def r_squared(self) -> float:
        tss = np.sum((self.y - self.y.mean()) ** 2)
        return 1 - self.resid @ self.resid / tss

    # inference --------------------------------------------------------------
    def cov(self, cov_type="classical", hac_lags=None) -> np.ndarray:
        return covariance(self.X, self.resid, cov_type, hac_lags)

    def se(self, cov_type="classical", hac_lags=None) -> np.ndarray:
        return np.sqrt(np.diag(self.cov(cov_type, hac_lags)))

    def t_test(self, null=0.0, cov_type="classical", hac_lags=None):
        se = self.se(cov_type, hac_lags)
        t = (self.beta - null) / se
        p = 2 * stats.t.sf(np.abs(t), self.df_resid)
        return t, p

    def conf_int(self, alpha=0.05, cov_type="classical", hac_lags=None) -> np.ndarray:
        se = self.se(cov_type, hac_lags)
        c = stats.t.ppf(1 - alpha / 2, self.df_resid)
        return np.column_stack([self.beta - c * se, self.beta + c * se])

    def f_test(self, Rmat, r=None, cov_type="classical", hac_lags=None):
        Rmat = np.atleast_2d(Rmat)
        r = np.zeros(Rmat.shape[0]) if r is None else np.asarray(r, float)
        return wald_test(self.beta, self.cov(cov_type, hac_lags), Rmat, r, self.df_resid)

    # diagnostics ------------------------------------------------------------
    def studentized_residuals(self) -> np.ndarray:
        """Internally studentised residuals e_i / (s sqrt(1 - h_ii))."""
        return self.resid / np.sqrt(self.sigma2 * (1 - self.leverage))

    def cooks_distance(self) -> np.ndarray:
        """D_i = r_i^2 / p * h_ii / (1 - h_ii): how far beta_hat moves when point i is dropped."""
        h = self.leverage
        return self.studentized_residuals() ** 2 / self.p * h / (1 - h)

    def summary(self, cov_type="classical", names=None) -> str:
        names = names or [f"b{j}" for j in range(self.p)]
        se = self.se(cov_type)
        t, pv = self.t_test(cov_type=cov_type)
        ci = self.conf_int(cov_type=cov_type)
        lines = [
            f"OLS ({self.method}), n={self.n}, p={self.p}, cov={cov_type}, R^2={self.r_squared():.4f}",
            f"{'':>8}{'coef':>10}{'se':>10}{'t':>9}{'p':>9}{'[0.025':>10}{'0.975]':>10}",
        ]
        for j, nm in enumerate(names):
            lines.append(
                f"{nm:>8}{self.beta[j]:>10.4f}{se[j]:>10.4f}{t[j]:>9.3f}{pv[j]:>9.4f}"
                f"{ci[j, 0]:>10.4f}{ci[j, 1]:>10.4f}"
            )
        return "\n".join(lines)
