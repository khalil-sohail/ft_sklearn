"""base classes and mixins for ML estimators

this module provides abstract base classes and mixins that define the standard
interface for all estimators in ft_sklearn, following scikit-learn conventions
"""

from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    """base class for all estimators
    
    provides common functionality for getting and setting estimator parameters.
    subclasses should inherit from this class to gain parameter management capabilities
    """

    def get_params(self):
        """get parameters for this estimator
        
        Returns:
            dict: parameter names mapped to their values. excludes parameters
                starting with underscore (private attributes)
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def set_params(self, **params):
        """set the parameters of this estimator
        
        Args:
            **params: estimator parameters to set
        
        Returns:
            BaseEstimator: returns self for method chaining
        """
        for k, v in params.items():
            setattr(self, k, v)
        return self


class ClassifierMixin:
    """mixin class for classifiers
    
    classes that inherit from this mixin should implement the predict method
    for classification tasks
    """

    @abstractmethod
    def predict(self, X):
        """predict class labels for samples in X
        
        Args:
            X: feature matrix of shape (n_samples, n_features)
        
        Returns:
            array-like: predicted class labels
        """
        ...


class RegressorMixin:
    """mixin class for regressors
    
    classes that inherit from this mixin should implement the predict method
    for regression tasks
    """

    @abstractmethod
    def predict(self, X):
        """predict target values for samples in X
        
        Args:
            X: feature matrix of shape (n_samples, n_features)
        
        Returns:
            array-like: predicted continuous values
        """
        ...


class TransformerMixin:
    """mixin class for transformers
    
    classes that inherit from this mixin should implement fit and transform
    """

    def fit_transform(self, X, y=None, **fit_params):
        """fit to data, then transform it
        
        Args:
            X: feature matrix
            y: target values (optional)
        """
        return self.fit(X, y, **fit_params).transform(X)
    