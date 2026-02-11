import numpy as np
import pytest
from ft_sklearn.linear_model import LogisticRegression
from ft_sklearn.preprocessing import StandardScaler

@pytest.fixture
def binary_data():
    X = np.array([
        [1, 2], [2, 3], [3, 3],
        [8, 9], [9, 9], [9, 8]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y

def test_logistic_regression(binary_data):
    X, y = binary_data
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000, eta0=0.1, random_state=42)
    model.fit(X_scaled, y)
    preds = model.predict(X_scaled)
    
    assert len(preds) == len(y)
    np.testing.assert_array_equal(preds, y)

def test_logistic_proba_shape(binary_data):
    X, y = binary_data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_scaled, y)
    probs = model.predict_proba(X_scaled)
    
    assert probs.shape == (6, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)