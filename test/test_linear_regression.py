import numpy as np
import pytest
from ft_sklearn.linear_model import LinearRegression

def test_linear_regression_simple():
    """test LinearRegression on a simple dataset"""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    assert model.coef_[0] == pytest.approx(2.0)
    
    predictions = model.predict(np.array([[5]]))
    assert predictions[0] == pytest.approx(10.0)

def test_linear_regression_with_intercept():
    """test LinearRegression with intercept"""
    X = np.array([[1], [2]])
    y = np.array([3, 5])

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    
    assert model.coef_[0] == pytest.approx(2.0)
    assert model.intercept_ == pytest.approx(1.0)