"""random forest classifier implementation"""

from ..base import BaseEstimator, ClassifierMixin


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    """a random forest classifier
    
    Attributes:
        estimators_ (list): collection of fitted sub-estimators
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        """initialize RandomForestClassifier"""
        pass

    def fit(self, X, y):
        """build a forest of trees from the training set (X, y)
        
        Args:
            X: training input samples
            y: target values
        """
        pass

    def predict(self, X):
        """predict class for X"""
        pass