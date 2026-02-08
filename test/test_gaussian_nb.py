import numpy as np
import pytest
from ft_sklearn.naive_bayes.GaussianNB import GaussianNB

@pytest.fixture
def binary_data():
    X = np.array([
        [1, 2], [2, 3], [3, 3],
        [8, 9], [9, 9], [9, 8]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y

@pytest.mark.skip(reason="Not implemented yet")
def test_gaussian_nb(binary_data):
    X, y = binary_data
    model = GaussianNB()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert np.all(preds == y)
