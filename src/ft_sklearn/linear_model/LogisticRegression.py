"""
logistic regression implementation
"""

from ..base import BaseEstimator, ClassifierMixin
from ..utils import check_X_y, check_X
import numpy as np

class LogisticRegression(BaseEstimator, ClassifierMixin):
    """logistic regression (aka logit, MaxEnt) classifier
    
    Attributes:
        coef_ (array): coefficient of the features in the decision function
        intercept_ (float): intercept (a.k.a. bias) added to the decision function
    """

    def __init__(self, penalty='l2', C=1.0, fit_intercept=True, max_iter=100, eta0=0.01, random_state=42):
        """initialize LogisticRegression"""
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.learning_rate = eta0
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """fit the model according to the given training data
        
        Args:
            X: training vector
            y: target vector
        """
        if self.random_state:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        self.coef_ = np.random.randn(n_features)
        self.intercept_ = 0.0

        for _ in range(self.max_iter):
            shuffled_X = np.random.permutation(n_samples)
            for row in shuffled_X:
                feature = X[row]
                target = y[row]

                pred = self.predict_proba(feature)[1]
                error = pred - target
                
                if self.penalty == "l1":
                    tax = (1 / self.C) * np.sign(self.coef_)
                elif self.penalty == "l2":
                    tax = (1 / self.C) * self.coef_
                else:
                    tax = 0

                # adjust the weight and bias
                self.coef_ -= self.learning_rate * ((error * feature) + (tax / n_samples))
                if self.fit_intercept == True:
                    self.intercept_ -= self.learning_rate * error

        return self

    def predict_proba(self, X):
        X = np.array(X)
        
        z = np.dot(X, self.coef_) + self.intercept_
        prob_1 = 1 / (1 + np.exp(-z))
        prob_0 = 1 - prob_1
        
        if X.ndim == 1:
            return np.array([prob_0, prob_1])
        return np.column_stack((prob_0, prob_1))

    def predict(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        probs = self.predict_proba(X)
        
        if X.ndim == 1:
            return 1 if probs[1] >= 0.5 else 0
        return (probs[:, 1] >= 0.5).astype(int)
    