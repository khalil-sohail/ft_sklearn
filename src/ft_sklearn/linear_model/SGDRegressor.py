# Initialize weights (w) and bias (b) to small random numbers (or zeros).
# Loop for a set number of epochs (passes through the dataset):
#     Shuffle the data (Important!).
#     Loop through every single data point (xi​,yi​):
#         Predict: Calculate y^​i​=w⋅xi​+b.
#         Calculate Error: e=y^​i​−yi​.
#         Calculate Gradient: How much did w contribute to the error?
#         Update Weights: wnew​=wold​−(learning_rate×gradient).

from ..base import BaseEstimator, RegressorMixin
from ..utils import check_X_y, check_X
import numpy as np

class ft_SGDRegressor((BaseEstimator, RegressorMixin)):
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # 1. Initialize weights (zeros is usually fine for simple regression)
        # 2. Loop over epochs
        # 3. Inside epoch, shuffle data
        # 4. Inside epoch, loop over every row
        # 5. Update weights using the formulas above
        pass

    def predict(self, X):
        # Same matrix math as before: y = X @ w + b
        pass