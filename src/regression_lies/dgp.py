"""
Data-generating processes (DGPs).

Every function returns (X, y, beta_true) for R independent datasets of size n:
    X : (R, n, p) design matrix with an intercept column
    y : (R, n)
    beta_true : (p,) the coefficients of the *model we fit*, i.e. the target
                that a well-behaved estimator should recover.

The baseline truth is   Y = 2 + 3 X + eps.

Each DGP breaks exactly one classical assumption, so its effect can be
measured in isolation. The assumptions:

    A1 Linearity          y = X beta + eps
    A2 Full rank          X'X invertible (no perfect multicollinearity)
    A3 Exogeneity         E[eps | X] = 0
    A4 Spherical errors   Var(eps | X) = sigma^2 I  (homoskedastic AND uncorrelated)
    A5 Normality          eps | X ~ N(0, sigma^2 I)   (needed only for exact small-n inference)
"""

from __future__ import annotations

import numpy as np

BETA = np.array([2.0, 3.0])


def _design(x: np.ndarray) -> np.ndarray:
    """Stack an intercept column next to the regressor."""
    return np.stack([np.ones_like(x), x], axis=-1)


def _ar1(rng, R, n, rho, burn=200):
    """Stationary Gaussian AR(1) with unit marginal variance: z_t = rho z_{t-1} + sqrt(1-rho^2) u_t."""
    u = rng.standard_normal((R, n + burn))
    z = np.empty_like(u)
    z[:, 0] = u[:, 0]
    s = np.sqrt(1 - rho**2)
    for t in range(1, n + burn):
        z[:, t] = rho * z[:, t - 1] + s * u[:, t]
    return z[:, burn:]


# ---------------------------------------------------------------------------
# All assumptions hold
# ---------------------------------------------------------------------------

def baseline(rng, R, n, beta=BETA, sigma=1.0):
    x = rng.standard_normal((R, n))
    eps = sigma * rng.standard_normal((R, n))
    X = _design(x)
    return X, X @ beta + eps, beta


# ---------------------------------------------------------------------------
# A3 broken: endogeneity
# ---------------------------------------------------------------------------

def omitted_variable(rng, R, n, beta=BETA, gamma=2.0, corr=0.5):
    """True model: y = 2 + 3x + gamma*w + u, with corr(x, w) = corr.

    We regress y on x only, so gamma*w hides in the error and is correlated with x.
    Probability limit of the slope: 3 + gamma * cov(x, w) / var(x) = 3 + gamma * corr.
    """
    x = rng.standard_normal((R, n))
    w = corr * x + np.sqrt(1 - corr**2) * rng.standard_normal((R, n))
    y = beta[0] + beta[1] * x + gamma * w + rng.standard_normal((R, n))
    return _design(x), y, beta


def measurement_error(rng, R, n, beta=BETA, me_sd=0.7):
    """We observe x* = x + m instead of x (classical errors-in-variables).

    The error term becomes u - beta1*m, which is correlated with x*.
    Probability limit of the slope: beta1 * var(x) / (var(x) + var(m))   (attenuation).
    """
    x = rng.standard_normal((R, n))
    y = beta[0] + beta[1] * x + rng.standard_normal((R, n))
    x_obs = x + me_sd * rng.standard_normal((R, n))
    return _design(x_obs), y, beta


def simultaneity(rng, R, n, beta=BETA, feedback=0.5):
    """Reverse causality: x also responds to the shock in y.

    x = z + feedback * eps.  Then cov(x, eps) = feedback * var(eps) > 0, and
    plim slope = beta1 + feedback / (1 + feedback^2).
    """
    z = rng.standard_normal((R, n))
    eps = rng.standard_normal((R, n))
    x = z + feedback * eps
    y = beta[0] + beta[1] * x + eps
    return _design(x), y, beta


# ---------------------------------------------------------------------------
# A4 broken: heteroskedasticity
# ---------------------------------------------------------------------------

def heteroskedastic(rng, R, n, beta=BETA, strength=1.0):
    """Var(eps | x) = 0.1 + strength * x^2   (noise grows away from the centre).

    This is the damaging kind: the noise is largest exactly where x has the most
    leverage on the slope, so the classical formula understates Var(slope).
    """
    x = rng.standard_normal((R, n))
    sd = np.sqrt(0.1 + strength * x**2)
    y = beta[0] + beta[1] * x + sd * rng.standard_normal((R, n))
    return _design(x), y, beta


# ---------------------------------------------------------------------------
# A4 broken: autocorrelation (time series)
# ---------------------------------------------------------------------------

def autocorrelated(rng, R, n, beta=BETA, rho_x=0.8, rho_e=0.8):
    """Both the regressor and the error are persistent AR(1) processes.

    Var(slope) is inflated by roughly (1 + rho_x rho_e) / (1 - rho_x rho_e),
    a factor the classical formula ignores. With rho = 0.8 that is about 4.6.
    """
    x = _ar1(rng, R, n, rho_x)
    eps = _ar1(rng, R, n, rho_e)
    y = beta[0] + beta[1] * x + eps
    return _design(x), y, beta


# ---------------------------------------------------------------------------
# A5 broken: non-Gaussian errors
# ---------------------------------------------------------------------------

def non_gaussian(rng, R, n, beta=BETA, dist="skewed"):
    """Errors with mean 0 and variance 1 but a non-normal shape.

    skewed     : centred chi-square(1)/sqrt(2)     (strong right skew, excess kurtosis 12)
    heavy_tail : Student t(3) scaled to unit variance (infinite fourth moment)
    cauchy     : standard Cauchy (no mean, no variance: every guarantee fails)
    normal     : reference case with the same skewed regressor

    The regressor is lognormal so that a few high-leverage points interact with
    the error shape, which is where small-sample inference suffers.
    """
    x = rng.lognormal(0.0, 1.0, (R, n))
    if dist == "skewed":
        eps = (rng.chisquare(1, (R, n)) - 1) / np.sqrt(2)
    elif dist == "heavy_tail":
        eps = rng.standard_t(3, (R, n)) / np.sqrt(3.0)
    elif dist == "cauchy":
        eps = rng.standard_cauchy((R, n))
    elif dist == "normal":
        eps = rng.standard_normal((R, n))
    else:
        raise ValueError(dist)
    y = beta[0] + beta[1] * x + eps
    return _design(x), y, beta


# ---------------------------------------------------------------------------
# A2 stressed: near multicollinearity
# ---------------------------------------------------------------------------

def collinear(rng, R, n, rho=0.9, beta=np.array([2.0, 3.0, -1.0])):
    """y = 2 + 3 x1 - 1 x2 + eps with corr(x1, x2) = rho.

    Still unbiased. Variance of each slope is inflated by VIF = 1 / (1 - rho^2).
    """
    x1 = rng.standard_normal((R, n))
    x2 = rho * x1 + np.sqrt(1 - rho**2) * rng.standard_normal((R, n))
    X = np.stack([np.ones_like(x1), x1, x2], axis=-1)
    y = X @ beta + rng.standard_normal((R, n))
    return X, y, beta


# ---------------------------------------------------------------------------
# Registry used by the experiments
# ---------------------------------------------------------------------------

def _skewed(rng, R, n, beta=BETA):
    return non_gaussian(rng, R, n, beta, "skewed")


def _heavy(rng, R, n, beta=BETA):
    return non_gaussian(rng, R, n, beta, "heavy_tail")


SCENARIOS = {
    "Valid assumptions": baseline,
    "Omitted variable": omitted_variable,
    "Measurement error": measurement_error,
    "Simultaneity": simultaneity,
    "Heteroskedastic": heteroskedastic,
    "Autocorrelated": autocorrelated,
    "Skewed errors": _skewed,
    "Heavy-tailed errors": _heavy,
}

# Which assumption each scenario violates (for tables and plots)
VIOLATES = {
    "Valid assumptions": "none",
    "Omitted variable": "A3 exogeneity",
    "Measurement error": "A3 exogeneity",
    "Simultaneity": "A3 exogeneity",
    "Heteroskedastic": "A4 homoskedasticity",
    "Autocorrelated": "A4 no autocorrelation",
    "Skewed errors": "A5 normality",
    "Heavy-tailed errors": "A5 normality",
}
