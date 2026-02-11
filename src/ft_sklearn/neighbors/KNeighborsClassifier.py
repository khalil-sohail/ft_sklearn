"""k-nearest neighbors classifier implementation"""

from ..base import BaseEstimator, ClassifierMixin
import numpy as np

class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """classifier implementing the k-nearest neighbors vote
    
    Attributes:
        X_train_ (array): training data
        y_train_ (array): training labels
    """

    def __init__(self, n_neighbors=5, metric='euclidean', weights="distance"):
        """initialize KNeighborsClassifier"""
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def _euclidean_distance(self, x1, x2):
        """Calculate the distance between two vectors."""
        return np.linalg.norm(x1 - x2)

    def _predict_single(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.n_neighbors]

        if self.weights == 'uniform':
            k_nearest_labels = self.y_train[k_indices]
            return np.bincount(k_nearest_labels).argmax()
        elif self.weights == 'distance':
            k_distances = np.array(distances)[k_indices]
            k_labels = self.y_train[k_indices]
            if np.any(k_distances == 0):
                weights = (k_distances == 0).astype(float)
            else:
                weights = 1.0 / (k_distances + 1e-10)
            
            class_scores = np.bincount(k_labels, weights=weights)
            return class_scores.argmax()

    def fit(self, X, y):
        """fit the k-nearest neighbors classifier from the training dataset
        
        Args:
            X: training data
            y: target values
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
        
    def predict_proba(self, X):
        X = np.array(X)
        all_probs = []
        
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_labels = self.y_train[k_indices]
            
            if self.weights == 'uniform':
                counts = np.bincount(k_labels, minlength=len(np.unique(self.y_train)))
                probs = counts / self.n_neighbors
                
            elif self.weights == 'distance':
                k_distances = distances[k_indices]
                weights = 1.0 / (k_distances + 1e-10)
                
                weighted_counts = np.bincount(k_labels, weights=weights, minlength=len(np.unique(self.y_train)))
                probs = weighted_counts / np.sum(weights)
                
            all_probs.append(probs)
            
        return np.array(all_probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
