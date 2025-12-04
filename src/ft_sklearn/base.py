import numpy as np

class BaseEstimator:
    """
    The parent class for all our ft_sklearn algorithms.
    It enforces the common structure and handles utility features.
    """
    def fit(self, X, y):
        # this function will be override.
        raise NotImplementedError("Subclass must implement abstract method")

    def predict(self, X):
        # this function will be override.
        raise NotImplementedError("Subclass must implement abstract method")

