"""principal component analysis (PCA) implementation"""

from ..base import BaseEstimator, TransformerMixin


class PCA(BaseEstimator, TransformerMixin):
    """principal component analysis (PCA)
    
    Attributes:
        components_ (array): principal axes in feature space
        explained_variance_ (array): the amount of variance explained by each component
        mean_ (array): per-feature empirical mean, estimated from the training set
    """

    def __init__(self, n_components=None):
        """initialize PCA"""
        pass

    def fit(self, X, y=None):
        """fit the model with X
        
        Args:
            X: training data
            y: ignored
        """
        pass

    def transform(self, X):
        """apply dimensionality reduction to X"""
        pass

    def inverse_transform(self, X):
        """transform data back to its original space"""
        pass