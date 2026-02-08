"""data validation utilities for estimator inputs

provides functions to validate and convert input data to expected formats
"""

import numpy as np


def check_X_y(X, y):
    """validate feature matrix and target vector
    
    ensures that X is a 2D array and y is a 1D array, and that they have
    matching numbers of samples. converts input to NumPy arrays
    
    Args:
        X (array-like): feature matrix of shape (n_samples, n_features)
        y (array-like): target vector of shape (n_samples,)
    
    Returns:
        tuple: validated (X, y) as NumPy arrays
    
    Raises:
        ValueError: if X is not 2D, y is not 1D, or they have different
            numbers of samples
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching rows")

    return X, y


def check_X(X):
    """validate feature matrix
    
    ensures that X is a 2D array. converts input to NumPy array
    
    Args:
        X (array-like): feature matrix of shape (n_samples, n_features)
    
    Returns:
        array: validated X as a NumPy array
    
    Raises:
        ValueError: if X is not 2D
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    return X
