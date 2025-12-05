import numpy as np

def check_X_y(X, y):
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
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    return X
