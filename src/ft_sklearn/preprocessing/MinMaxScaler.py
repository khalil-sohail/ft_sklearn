"""Min-Max normalization preprocessing scaler.

Scales features to a fixed range, typically [0, 1].
"""

import numpy as np


class MinMaxScaler:
    """Scale features to a fixed range, typically [0, 1].
    
    The MinMax scaling formula for each feature is:
        X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
    
    Attributes:
        feature_range (tuple): Desired range of transformed data (min, max).
        copy (bool): If True, copy input data before transformation.
        clip (bool): If True, clip transformed values to the feature range.
        data_min_ (array): Minimum value for each feature in training set.
        data_max_ (array): Maximum value for each feature in training set.
        data_range_ (array): Range of each feature (max - min).
        scale_ (array): Scaling factor for each feature.
        min_ (array): Offset for each feature.
    """

    def __init__(self, feature_range=(0, 1), copy=True, clip=False):
        """Initialize MinMaxScaler.
        
        Args:
            feature_range (tuple, optional): Desired range (min, max). Default is (0, 1).
            copy (bool, optional): If True, copy input data. Default is True.
            clip (bool, optional): If True, clip to feature range. Default is False.
        """
        self.copy = copy
        self.feature_range = feature_range
        self.clip = clip

        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.scale_ = None
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        """Compute min and max values for later scaling.
        
        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
        
        Returns:
            MinMaxScaler: Returns self for method chaining.
        """
        if not self.copy and X.dtype != float:
            X[:] = X.astype(float)
        elif self.copy:
            X = X.astype(float)

        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        (min, max) = self.feature_range
        self.scale_ = (max - min) / self.data_range_
        self.min_ = min - (self.data_min_ * self.scale_)

        return self

    def transform(self, X):
        """Scale features to the specified range.
        
        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
        
        Returns:
            array: Transformed feature matrix of same shape as X.
        """
        if not self.copy and X.dtype != float:
            X[:] = X.astype(float)
        elif self.copy:
            X = X.astype(float)

        X *= self.scale_
        X += self.min_
        if self.clip:
            np.clip(X, self.feature_range[0], self.feature_range[1], out=X)

        return X

    def fit_transform(self, X):
        """Fit to data, then transform it.
        
        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
        
        Returns:
            array: Transformed feature matrix of same shape as X.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, transformed_X):
        """Scale back the data to the original representation.
        
        Args:
            transformed_X (array-like): Transformed feature matrix of shape
                (n_samples, n_features).
        
        Returns:
            array: Original feature matrix.
        """
        if not self.copy and transformed_X.dtype != float:
            transformed_X[:] = transformed_X.astype(float)
        elif self.copy:
            transformed_X = transformed_X.astype(float)

        transformed_X /= self.scale_
        transformed_X -= self.min_

        return transformed_X
