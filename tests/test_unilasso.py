import pytest
import numpy as np
import pandas as pd
from unilasso import fit_unilasso, cv_unilasso, simulate_cox_data, simulate_gaussian_data, predict, extract_cv




def _check_result(result, p, family):
    assert hasattr(result, "coefs")
    assert hasattr(result, "intercept")
    assert hasattr(result, "_gamma")
    assert hasattr(result, "_beta")

    assert result.coefs.shape == (p,)
    assert type(result.intercept) == np.ndarray
    assert result._gamma.shape == (p,)
    assert result._beta.shape == (p,)

    if family == "cox":
        # Cox model has an additional intercept term
        assert result.intercept == 0
        assert result._gamma_intercept == 0
        assert np.all(result._beta_intercepts == 0)




@pytest.mark.parametrize("family", ["gaussian", "binomial", "cox"])
def test_different_families(family):
    if family == "cox":
        X, y = simulate_cox_data(n=100, p=5, seed=123)
    else:
        X = np.random.randn(100, 5)
        if family == "binomial":
            y = np.random.randint(2, size=100)
        else:
            y = np.random.randn(100)
    result = fit_unilasso(X, y, family=family, lmdas=[0.1])
    _check_result(result, 5, family)


def test_default_lmda():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    result = fit_unilasso(X, y, family="gaussian")


def test_fit_gaussian():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    result = fit_unilasso(X, y, family="gaussian", lmdas=[0.1])

    X[:, 0] = 1
    result = fit_unilasso(X, y, family="gaussian", lmdas=[0.1])
    assert result.coefs[0] == 0 # First coefficient should be zero
    
    _check_result(result, 5, "gaussian")



def test_fit_binomial():
    X = np.random.rand(100, 5)
    y = np.random.randint(2, size=100)
    result = fit_unilasso(X, y, family="binomial", lmdas=[0.1])
    
    _check_result(result, 5, "binomial")


def test_fit_cox():
    X, y = simulate_cox_data(n=100, p=5, seed=123)
    result = fit_unilasso(X, y, family="cox", lmdas=[0.1])
    _check_result(result, 5, "cox")

    y = pd.DataFrame(y, columns=['time', 'status'])
    result = fit_unilasso(X, y, family="cox", lmdas=[0.1])
    _check_result(result, 5, "cox")


    y = pd.DataFrame(y, columns=['time', 'trend'])
    with pytest.raises(ValueError):
        fit_unilasso(X, y, family="cox", lmdas=[0.1])

    
def test_zero_variance():
    X = np.random.rand(100, 5)
    X[:, 0] = 1
    y = np.random.rand(100)
    result = fit_unilasso(X, y, family="gaussian", lmdas=[0.1])

    _check_result(result, 5, "gaussian")
    assert result._gamma[0] == 0 # First feature has zero variance
    assert result._beta[0] == 0 # First feature has zero variance

    # Model should throw error if all features have zero variance
    X = np.zeros((100, 5))
    with pytest.raises(ValueError):
        fit_unilasso(X, y, family="cox", lmdas=[0.1])



def test_input_validation():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    with pytest.raises(ValueError):
        fit_unilasso(X, y, family='invalid_family', lmdas=[0.1])


def _check_cv_result(result, p, family):
    num_lmdas = len(result.lmdas)
    assert hasattr(result, "cv_plot")
    assert hasattr(result, "best_idx")
    assert isinstance(result.best_idx, int)
    assert isinstance(result.lmdas, np.ndarray)
    assert result.coefs.shape == (num_lmdas, p)


def test_cv_unilasso():
    X, y = simulate_gaussian_data(n=100, p=5, seed=123)

    # Test extra arguments for cv_unilasso
    with pytest.raises(TypeError) as excinfo:
        cv_unilasso(X, y, family='gaussian', lmdas=[0.1])
        
    result = cv_unilasso(X, y, family="gaussian")
    _check_cv_result(result, 5, "gaussian")

    extracted_fit = extract_cv(result)
    assert extracted_fit.coefs.shape == (5, )
    assert type(extracted_fit.intercept) == np.float64


def test_predict():
    X, y = simulate_gaussian_data(n=100, p=5)
    result = fit_unilasso(X, y, family="gaussian", lmdas=[0.1, 0.2])
    y_pred = predict(result, X)
    assert y_pred.shape == (100, 2)

    y_pred_lmda_1 = predict(result, X, lmda_idx=0)
    assert y_pred_lmda_1.shape == (100, )

    # Test invalid dimension
    X, y = simulate_cox_data(n=100, p=5)
    result = fit_unilasso(X, y, family="cox", lmdas=[0.1, 0.2])
    X_test, y_test = simulate_cox_data(n=100, p=10)
    with pytest.raises(AssertionError):
        y_pred = predict(result, X_test)  # X_test has different number of features

@pytest.mark.parametrize("lmda", [0.05, 0.01])
def test_loo_false_is_the_lasso_with_univariate_penalty_factors(lmda):
    # loo=False solves min (1/2n)||y - b0 - X coef||^2 + lmda sum_j |coef_j| / |beta_j|
    # with sign(coef_j) in {0, sign(beta_j)}, beta_j the univariate slopes:
    # with centered data, beta_j c_j'r = n lmda where coef_j != 0, and <= n lmda elsewhere
    X, y = simulate_gaussian_data(n=300, p=8)
    n = X.shape[0]
    fit = fit_unilasso(X, y, lmdas=lmda, loo=False)
    coef = np.ravel(fit.coefs)
    Xc, yc = X - X.mean(0), y - y.mean()
    beta = Xc.T @ yc / np.sum(Xc**2, axis=0)
    np.testing.assert_allclose(np.ravel(fit._beta), beta, rtol=1e-8)
    assert np.all(coef * np.sign(beta) >= 0)
    score = beta * (Xc.T @ (yc - Xc @ coef))
    active = coef != 0
    assert active.any()
    np.testing.assert_allclose(score[active], n * lmda, rtol=1e-3)
    assert np.all(score[~active] <= n * lmda * (1 + 1e-3))
    # the intercept is the least squares one given coef
    np.testing.assert_allclose(np.ravel(fit.intercept)[0], y.mean() - X.mean(0) @ coef, atol=1e-6)


def test_loo_false_differs_from_loo():
    X, y = simulate_gaussian_data(n=200, p=8)
    with_loo = np.ravel(fit_unilasso(X, y, lmdas=0.01).coefs)
    without = np.ravel(fit_unilasso(X, y, lmdas=0.01, loo=False).coefs)
    assert not np.allclose(with_loo, without)


@pytest.mark.parametrize("family", ["binomial", "cox"])
def test_loo_false_other_families(family):
    X, y = simulate_gaussian_data(n=200, p=5)
    if family == "binomial":
        y = (y > np.median(y)).astype(float)
    else:
        X, y = simulate_cox_data(n=200, p=5)
    fit = fit_unilasso(X, y, family=family, lmdas=0.01, loo=False)
    _check_result(fit, X.shape[1], family)


def test_cv_loo_false():
    X, y = simulate_gaussian_data(n=200, p=8)
    cv = cv_unilasso(X, y, loo=False, seed=0)
    assert cv.best_lmda in cv.lmdas
    assert np.all(np.isfinite(extract_cv(cv).coefs))
