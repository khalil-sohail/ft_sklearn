import numpy as np

class StandardScaler():
    """
    Docstring for StandardScaler
    """
    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
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
        """
        Docstring for transform
        
        :param self: Description
        :param X: Description
        """
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
        return self.fit(X).transform(X)
    
    def inverse_transform(self, transformed_X):
        """
        Docstring for inverse_transform
        
        :param self: Description
        :param transformed_X: Description
        """
        if self.copy:
            transformed_X = transformed_X.copy()

        transformed_X = transformed_X.astype(float)

        if self.with_std and self.scale_ is not None:
            transformed_X *= self.scale_
        if self.with_mean and self.mean_ is not None:
            transformed_X += self.mean_

        return transformed_X

