"""When Regression Lies: a computational study of OLS assumptions and inference."""

from .ols import OLS, fit_batch, standard_errors, covariance, leverage, hat_matrix, solve, wald_test
from . import dgp, montecarlo

__all__ = [
    "OLS", "fit_batch", "standard_errors", "covariance", "leverage",
    "hat_matrix", "solve", "wald_test", "dgp", "montecarlo",
]
