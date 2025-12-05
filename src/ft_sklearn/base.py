from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    def get_params(self):
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class ClassifierMixin:
    @abstractmethod
    def predict(self, X):
        ...

class RegressorMixin:
    @abstractmethod
    def predict(self, X):
        ...
    