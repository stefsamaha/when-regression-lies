"""
Every quantity implemented from scratch is checked against statsmodels,
plus the algebraic identities that define a least-squares projection.
"""

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

from regression_lies import OLS, fit_batch, standard_errors, hat_matrix, dgp
from regression_lies.ols import SOLVERS, covariance


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    n = 120
    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.uniform(-2, 2, n)])
    y = X @ np.array([1.0, 2.0, -0.5]) + rng.standard_normal(n) * (0.5 + np.abs(X[:, 1]))
    return X, y


@pytest.fixture
def ref(data):
    X, y = data
    return sm.OLS(y, X).fit()


@pytest.mark.parametrize("method", SOLVERS)
def test_coefficients_all_solvers(data, ref, method):
    X, y = data
    np.testing.assert_allclose(OLS(method).fit(X, y).beta, ref.params, rtol=1e-10)


def test_classical_inference(data, ref):
    X, y = data
    m = OLS().fit(X, y)
    np.testing.assert_allclose(m.se(), ref.bse, rtol=1e-10)
    t, p = m.t_test()
    np.testing.assert_allclose(t, ref.tvalues, rtol=1e-10)
    np.testing.assert_allclose(p, ref.pvalues, rtol=1e-8)
    np.testing.assert_allclose(m.conf_int(), ref.conf_int(), rtol=1e-10)
    np.testing.assert_allclose(m.r_squared(), ref.rsquared, rtol=1e-12)


@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3"])
def test_robust_se(data, cov_type):
    X, y = data
    ref = sm.OLS(y, X).fit(cov_type=cov_type)
    np.testing.assert_allclose(OLS().fit(X, y).se(cov_type), ref.bse, rtol=1e-10)


@pytest.mark.parametrize("lags", [1, 4, 10])
def test_newey_west(data, lags):
    X, y = data
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})
    np.testing.assert_allclose(OLS().fit(X, y).se("HAC", hac_lags=lags), ref.bse, rtol=1e-10)


def test_f_test(data, ref):
    X, y = data
    m = OLS().fit(X, y)
    Rm = np.array([[0, 1, 0], [0, 0, 1]])
    F, p = m.f_test(Rm)
    np.testing.assert_allclose(F, ref.fvalue, rtol=1e-10)
    np.testing.assert_allclose(p, ref.f_pvalue, rtol=1e-8)


def test_diagnostics(data, ref):
    X, y = data
    m = OLS().fit(X, y)
    infl = OLSInfluence(ref)
    np.testing.assert_allclose(m.leverage, infl.hat_matrix_diag, rtol=1e-10)
    np.testing.assert_allclose(m.studentized_residuals(), infl.resid_studentized_internal, rtol=1e-10)
    np.testing.assert_allclose(m.cooks_distance(), infl.cooks_distance[0], rtol=1e-10)


def test_projection_identities(data):
    """H is the orthogonal projector onto col(X)."""
    X, y = data
    H = hat_matrix(X)
    m = OLS().fit(X, y)
    np.testing.assert_allclose(H, H.T, atol=1e-12)                 # symmetric
    np.testing.assert_allclose(H @ H, H, atol=1e-12)               # idempotent
    np.testing.assert_allclose(np.trace(H), X.shape[1], atol=1e-10)  # trace = rank = p
    np.testing.assert_allclose(H @ y, m.fitted, atol=1e-10)        # H y = y_hat
    np.testing.assert_allclose(X.T @ m.resid, 0, atol=1e-9)        # residuals orthogonal to every column
    np.testing.assert_allclose(m.fitted @ m.resid, 0, atol=1e-9)   # and to the fitted values
    np.testing.assert_allclose(m.resid.sum(), 0, atol=1e-9)        # intercept => residuals sum to 0


def test_batch_matches_single():
    rng = np.random.default_rng(0)
    X, y, _ = dgp.heteroskedastic(rng, 7, 50)
    fit = fit_batch(X, y)
    for cov_type in ["classical", "HC0", "HC3", "HAC"]:
        se = standard_errors(X, fit, cov_type)
        for r in range(7):
            m = OLS().fit(X[r], y[r])
            np.testing.assert_allclose(fit.beta[r], m.beta, rtol=1e-10)
            np.testing.assert_allclose(se[r], m.se(cov_type), rtol=1e-10)


def test_covariances_positive_definite():
    """Sandwich estimators must stay positive definite (the Bartlett kernel guarantees it for HAC)."""
    rng = np.random.default_rng(1)
    X, y, _ = dgp.autocorrelated(rng, 20, 80)
    fit = fit_batch(X, y)
    for cov_type in ["HC0", "HC3", "HAC"]:
        V = covariance(X, fit.resid, cov_type)
        assert np.all(np.linalg.eigvalsh(V) > 0)
