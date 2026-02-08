"""k-nearest neighbors classifier implementation"""

from ..base import BaseEstimator, ClassifierMixin


class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """classifier implementing the k-nearest neighbors vote
    
    Attributes:
        X_train_ (array): training data
        y_train_ (array): training labels
    """

    def __init__(self, n_neighbors=5, metric='euclidean'):
        """initialize KNeighborsClassifier"""
        pass

    def fit(self, X, y):
        """fit the k-nearest neighbors classifier from the training dataset
        
        Args:
            X: training data
            y: target values
        """
        pass

    def predict(self, X):
        """predict the class labels for the provided data"""
        pass