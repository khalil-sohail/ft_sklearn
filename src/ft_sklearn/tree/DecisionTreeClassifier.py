"""decision tree classifier implementation"""

from ..base import BaseEstimator, ClassifierMixin


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """a decision tree classifier
    
    Attributes:
        tree_ (object): the underlying tree structure
    """

    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2):
        """initialize DecisionTreeClassifier"""
        pass

    def fit(self, X, y):
        """build a decision tree classifier from the training set (X, y)
        
        Args:
            X: training input samples
            y: target values
        """
        pass

    def predict(self, X):
        """predict class or regression value for X"""
        pass