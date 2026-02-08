"""support vector classification implementation"""

from ..base import BaseEstimator, ClassifierMixin


class SVC(BaseEstimator, ClassifierMixin):
    """c-support vector classification
    
    Attributes:
        support_vectors_ (array): support vectors
        dual_coef_ (array): coefficients of the support vector in the decision function
    """

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale'):
        """initialize SVC"""
        pass

    def fit(self, X, y):
        """fit the SVM model according to the given training data
        
        Args:
            X: training vectors
            y: target values
        """
        pass

    def predict(self, X):
        """perform classification on samples in X"""
        pass