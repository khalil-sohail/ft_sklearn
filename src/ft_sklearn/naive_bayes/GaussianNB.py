"""gaussian naive bayes implementation"""

from ..base import BaseEstimator, ClassifierMixin


class GaussianNB(BaseEstimator, ClassifierMixin):
    """gaussian naive bayes (GaussianNB)
    
    Attributes:
        class_prior_ (array): probability of each class
        theta_ (array): mean of each feature per class
        var_ (array): variance of each feature per class
    """

    def __init__(self):
        """initialize GaussianNB"""
        pass

    def fit(self, X, y):
        """fit Gaussian Naive Bayes according to X, y
        
        Args:
            X: training vectors
            y: target values
        """
        pass

    def predict(self, X):
        """perform classification on an array of test vectors X"""
        pass