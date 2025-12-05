from ..base import BaseEstimator, RegressorMixin
from ..utils import check_X_y, check_X
import numpy as np


class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        self.coef_, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.intercept_ = self.coef_[0]
        return self

    def predict(self, X):
        X = check_X(X)

        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        return (X @ self.coef_)
