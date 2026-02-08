import numpy as np
import pytest
from ft_sklearn.linear_model.SGDRegressor import ft_SGDRegressor as SGDRegressor

@pytest.mark.skip(reason="Not implemented yet")
def test_sgd_regressor():
    """test SGDRegressor on a simple dataset"""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = SGDRegressor()
    model.fit(X, y)
    
    predictions = model.predict(np.array([[5]]))
    assert predictions[0] == pytest.approx(10.0)
