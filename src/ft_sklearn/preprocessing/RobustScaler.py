import numpy as np

class RobustScaler():
    """
    Docstring for RobustScaler
    """
    def __init__(self, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True, unit_variance=False):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        self.unit_variance = unit_variance

        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Docstring for fit
        
        :param self: Description
        :param X: Description
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
        """
        Docstring for transform
        
        :param self: Description
        :param X: Description
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
        return self.fit(X).transform(X)
    
    def inverse_transform(self, transformed_X):
        """
        Docstring for inverse_transform
        
        :param self: Description
        :param transformed_X: Description
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
