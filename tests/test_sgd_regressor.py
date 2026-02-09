import numpy as np
import pytest
from ft_sklearn.linear_model import SGDRegressor
from ft_sklearn.preprocessing import StandardScaler as ft_StandardScaler
from sklearn.preprocessing import StandardScaler

def test_sgd_regressor():
    """test SGDRegressor on a simple dataset"""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = SGDRegressor(eta0=0.01, max_iter=500, random_state=42)
    scaler = ft_StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    X_new_scaled = scaler.transform([[5]])
    predictions = model.predict(X_new_scaled)

    assert predictions[0] == pytest.approx(10.0)
