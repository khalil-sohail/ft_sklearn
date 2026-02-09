"""standardization preprocessing scaler

scales features by removing the mean and scaling to unit variance
"""

import numpy as np


class StandardScaler:
    """standardize features by removing mean and scaling to unit variance
    
    the standard score of a sample x is calculated as:
        z = (x - mean) / std
    
    Attributes:
        copy (bool): if True, copy input data before transformation
        with_mean (bool): if True, center the data before scaling
        with_std (bool): if True, scale the data to unit variance
        mean_ (array): the mean value for each feature in the training set
        scale_ (array): per feature standard deviation
    """

    def __init__(self, copy=True, with_mean=True, with_std=True):
        """initialize StandardScaler
        
        Args:
            copy (bool, optional): if True, copy input data. default is True
            with_mean (bool, optional): if True, center the data. default is True
            with_std (bool, optional): if True, scale to unit variance. default is True
        """
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """compute mean and standard deviation for later scaling
        
        Args:
            X (array-like): feature matrix of shape (n_samples, n_features)
        
        Returns:
            StandardScaler: returns self for method chaining
        """
        if not self.copy and X.dtype != float:
            X[:] = X.astype(float)
        elif self.copy:
            X = X.astype(float)

        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = None

        if self.with_std:
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0.0] = 1.0
        else:
            self.scale_ = None

        return self

    def transform(self, X):
        """perform standardization by centering and scaling
        
        Args:
            X (array-like): feature matrix of shape (n_samples, n_features)
        
        Returns:
            array: transformed feature matrix of same shape as X
        """
        X = np.array(X)
        if self.copy:
            X = X.copy()

        if not self.copy and X.dtype != float:
            X[:] = X.astype(float)
        elif self.copy:
            X = X.astype(float)

        if self.with_mean and self.mean_ is not None:
            X -= self.mean_
        if self.with_std and self.scale_ is not None:
            X /= self.scale_

        return X

    def fit_transform(self, X):
        """fit to data, then transform it
        
        Args:
            X (array-like): feature matrix of shape (n_samples, n_features)
        
        Returns:
            array: transformed feature matrix of same shape as X
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, transformed_X):
        """scale back the data to the original representation
        
        Args:
            transformed_X (array-like): transformed feature matrix of shape
                (n_samples, n_features)
        
        Returns:
            array: original feature matrix
        """
        if self.copy:
            transformed_X = transformed_X.copy()

        transformed_X = transformed_X.astype(float)

        if self.with_std and self.scale_ is not None:
            transformed_X *= self.scale_
        if self.with_mean and self.mean_ is not None:
            transformed_X += self.mean_

        return transformed_X
