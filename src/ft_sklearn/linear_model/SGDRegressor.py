"""
stochastic gradient descent regressor

implements linear regression using the stochastic gradient descent optimization
algorithm. this regressor updates coefficients incrementally as it processes
individual samples, making it suitable for large datasets
"""

from ..base import BaseEstimator, RegressorMixin
from ..utils import check_X_y, check_X
import numpy as np

class SGDRegressor(BaseEstimator, RegressorMixin):
    """stochastic gradient descent regressor
    
    fits a linear model using stochastic gradient descent. useful for large-scale
    learning as it processes one sample at a time and updates the model incrementally
    
    Attributes:
        lr (float): learning rate that controls the step size for weight updates
        max_iter (int): number of passes over the training dataset
        coef_ (array): (weights) estimated coefficients for the linear regression problem
        intercept_ (float): (bias) independent term in the linear model
    """

    def __init__(self, eta0=0.01, max_iter=1000, random_state=42):
        """initialize SGDRegressor
        
        Args:
            eta0 (float, optional): learning rate for gradient descent.
                default is 0.01
            max_iter (int, optional): number of passes over training data.
                default is 1000
            random_state (int, optinal): 
        """
        self.max_iter = max_iter
        self.learning_rate = eta0
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        """fit the SGD regressor
        
        Args:
            X (array-like): training feature matrix of shape (n_samples, n_features)
            y (array-like): target values of shape (n_samples,)
        Returns:
            ft_SGDRegressor: returns self for method chaining
        Raises:
            ValueError: if X or y have incorrect dimensions or shapes
        """
        if self.random_state:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape 
        self.coef_ = np.random.randn(n_features) 

        for i in range(self.max_iter):
            shuffled_X = np.random.permutation(n_samples)
            for row in shuffled_X:
                feature = X[row]
                target = y[row]

                pred = np.dot(feature, self.coef_) + self.intercept_
                error = pred - target

                # adjust the weight and bias
                self.coef_ = self.coef_ - (self.learning_rate * error * feature)
                self.intercept_ = self.intercept_ - (self.learning_rate * error)

        return self

    def predict(self, X):
        """predict using the SGD regressor
        
        Args:
            X (array-like): feature matrix of shape (n_samples, n_features)
        Returns:
            array: predicted values of shape (n_samples,)
        Raises:
            ValueError: if X has incorrect dimensions
        """
        return np.dot(X, self.coef_) + self.intercept_