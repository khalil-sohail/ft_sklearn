"""logistic regression implementation"""

from ..base import BaseEstimator, ClassifierMixin


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """logistic regression (aka logit, MaxEnt) classifier
    
    Attributes:
        coef_ (array): coefficient of the features in the decision function
        intercept_ (float): intercept (a.k.a. bias) added to the decision function
    """

    def __init__(self, penalty='l2', C=1.0, fit_intercept=True, max_iter=100, learning_rate=0.01):
        """initialize LogisticRegression"""
        pass

    def fit(self, X, y):
        """fit the model according to the given training data
        
        Args:
            X: training vector
            y: target vector
        """
        pass

    def predict(self, X):
        """predict class labels for samples in X"""
        pass