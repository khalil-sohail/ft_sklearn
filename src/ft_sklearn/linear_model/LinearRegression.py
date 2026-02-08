"""linear regression model using ordinary least squares

this module implements linear regression using the closed-form solution
computed via the least squares method
"""

from ..base import BaseEstimator, RegressorMixin
from ..utils import check_X_y, check_X
import numpy as np


class LinearRegression(BaseEstimator, RegressorMixin):
    """ordinary least squares linear regression
    
    fits a linear model with coefficients w and intercept b to minimize
    the residual sum of squares between observed and predicted values
    
    Attributes:
        fit_intercept (bool): whether to calculate the intercept for this model
        coef_ (array): estimated coefficients for the linear regression problem
        intercept_ (float): independent term in the linear model
    """

    def __init__(self, fit_intercept=True):
        """initialize LinearRegression
        
        Args:
            fit_intercept (bool, optional): whether to fit an intercept term.
                default is True
        """
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """fit linear model
        
        Args:
            X (array-like): training feature matrix of shape (n_samples, n_features)
            y (array-like): target values of shape (n_samples,)
        
        Returns:
            LinearRegression: returns self for method chaining
        
        Raises:
            ValueError: if X or y have incorrect dimensions or shapes
        """
        X, y = check_X_y(X, y)

        if self.fit_intercept:
            X_b = np.c_[np.ones(X.shape[0]), X]
            coef, *_ = np.linalg.lstsq(X_b, y, rcond=None)
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, X):
        """predict using the linear model
        
        Args:
            X (array-like): feature matrix of shape (n_samples, n_features)
        
        Returns:
            array: predicted values of shape (n_samples,)
        
        Raises:
            ValueError: if X has incorrect dimensions
        """
        X = check_X(X)

        return (X @ self.coef_ + self.intercept_)
