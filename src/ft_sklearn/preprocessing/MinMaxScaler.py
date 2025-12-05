import numpy as np

class MinMaxScaler():
    """
    Docstring for MinMaxScaler
    """
    def __init__(self, feature_range=(0, 1), copy=True, clip=False):
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
        """
        Docstring for fit
        
        :param self: Description
        :param X: Description
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
        """
        Docstring for transform
        
        :param self: Description
        :param X: Description
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

        transformed_X /= self.scale_
        transformed_X -= self.min_

        return transformed_X
