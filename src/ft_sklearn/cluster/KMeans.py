"""k-means clustering implementation"""

from ..base import BaseEstimator, TransformerMixin


class KMeans(BaseEstimator, TransformerMixin):
    """k-means clustering
    
    Attributes:
        cluster_centers_ (array): coordinates of cluster centers
        labels_ (array): labels of each point
        inertia_ (float): sum of squared distances of samples to their closest cluster center
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4):
        """initialize KMeans"""
        pass

    def fit(self, X, y=None):
        """compute k-means clustering
        
        Args:
            X: training instances to cluster
            y: ignored
        """
        pass

    def predict(self, X):
        """predict the closest cluster each sample in X belongs to"""
        pass

    def transform(self, X):
        """transform X to a cluster-distance space"""
        pass