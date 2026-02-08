"""gradient boosting classifier implementation"""

from ..base import BaseEstimator, ClassifierMixin


class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """gradient boosting for classification
    
    Attributes:
        estimators_ (list): the collection of fitted sub-estimators
    """

    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3):
        """initialize GradientBoostingClassifier"""
        pass

    def fit(self, X, y):
        """fit the gradient boosting model
        
        Args:
            X: training input samples
            y: target values
        """
        pass

    def predict(self, X):
        """predict class for X"""
        pass