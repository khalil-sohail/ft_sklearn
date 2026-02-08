"""robust scaling preprocessing scaler

scales features using statistics that are robust to outliers (median and
interquartile range)
"""

import numpy as np


class RobustScaler:
    """scale features using statistics robust to outliers
    
    uses the median and interquartile range (IQR) instead of mean and standard
    deviation, making it resilient to outliers
    
    the robust scaling formula is:
        X_scaled = (X - median) / IQR
    
    Attributes:
        with_centering (bool): if True, center data to the median
        with_scaling (bool): if True, scale the data to the IQR
        quantile_range (tuple): quantile range for IQR calculation
        copy (bool): if True, copy input data before transformation
        unit_variance (bool): if True, scale to unit variance
        center_ (array): median value for each feature in training set
        scale_ (array): interquartile range for each feature
    """

    def __init__(self, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True, unit_variance=False):
        """initialize RobustScaler
        
        Args:
            with_centering (bool, optional): if True, center to median. default is True
            with_scaling (bool, optional): if True, scale to IQR. default is True
            quantile_range (tuple, optional): quantile range (Q1, Q3). default is (25.0, 75.0)
            copy (bool, optional): if True, copy input data. default is True
            unit_variance (bool, optional): if True, scale to unit variance. default is False
        """
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        self.unit_variance = unit_variance

        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        """compute median and IQR for later scaling
        
        Args:
            X (array-like): feature matrix of shape (n_samples, n_features)
        
        Returns:
            RobustScaler: returns self for method chaining
        """
        if not self.copy and X.dtype != float:
            X[:] = X.astype(float)
        elif self.copy:
            X = X.astype(float)

        if self.with_centering:
            self.center_ = np.nanmedian(X, axis=0)
        else:
            self.center_ = None

        Q1, Q2 = self.quantile_range
        val_min = np.percentile(X, Q1, axis=0)
        val_max = np.percentile(X, Q2, axis=0)
        self.scale_ = val_max - val_min

        if self.unit_variance and self.with_scaling:
            self.scale_ /= 1.349
            self.scale_[self.scale_ == 0.0] = 1.0
        elif self.with_scaling is False:
            self.scale_ = None

        return self

    def transform(self, X):
        """center and scale features using robust statistics
        
        Args:
            X (array-like): feature matrix of shape (n_samples, n_features)
        
        Returns:
            array: transformed feature matrix of same shape as X
        
        Raises:
            ValueError: if scaler has not been fitted yet
        """
        if not self.copy and X.dtype != float:
            X[:] = X.astype(float)
        elif self.copy:
            X = X.astype(float)

        if self.with_centering:
            if self.center_ is None:
                raise ValueError("Scaler has not been fitted yet.")
            X -= self.center_

        if self.with_scaling:
            if self.scale_ is None:
                raise ValueError("Scaler has not been fitted yet.")
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
        
        Raises:
            ValueError: if scaler has not been fitted yet
        """
        if not self.copy and transformed_X.dtype != float:
            transformed_X[:] = transformed_X.astype(float)
        elif self.copy:
            transformed_X = transformed_X.astype(float)

        if self.with_scaling:
            if self.scale_ is None:
                raise ValueError("Scaler has not been fitted yet.")
            transformed_X *= self.scale_

        if self.with_centering:
            if self.center_ is None:
                raise ValueError("Scaler has not been fitted yet.")
            transformed_X += self.center_

        return transformed_X
