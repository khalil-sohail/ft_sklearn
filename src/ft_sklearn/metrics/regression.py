"""regression metrics for model evaluation

provides functions to compute common regression performance metrics
"""

import numpy as np
import math


def mean_squared_error(y_true, y_pred):
    """mean squared error (MSE) regression loss
    
    calculated as the average of squared differences between predicted and
    actual values. gives more weight to large errors
    
    Formula:
        MSE = (1/n) * sum((y_true - y_pred)^2)
    
    Args:
        y_true (array-like): ground truth target values
        y_pred (array-like): estimated target values
    
    Returns:
        float: mean squared error value. lower is better
    """
    n_samples = len(y_true)
    s = sum(((y_true[i] - y_pred[i])**2) for i in range(0, n_samples))
    return (1/n_samples) * s


def mean_absolute_error(y_true, y_pred):
    """mean absolute error (MAE) regression loss
    
    calculated as the average of absolute differences between predicted and
    actual values. more robust to outliers than MSE
    
    Formula:
        MAE = (1/n) * sum(|y_true - y_pred|)
    
    Args:
        y_true (array-like): ground truth target values
        y_pred (array-like): estimated target values
    
    Returns:
        float: mean absolute error value. lower is better
    """
    n_samples = len(y_true)
    s = sum(abs(y_true[i] - y_pred[i]) for i in range(0, n_samples))
    return (1/n_samples) * s


def root_mean_squared_error(y_true, y_pred):
    """root mean squared error (RMSE) regression loss
    
    calculated as the square root of MSE. penalizes large errors more than MAE
    while being in the same units as the target variable
    
    Formula:
        RMSE = sqrt((1/n) * sum((y_true - y_pred)^2))
    
    Args:
        y_true (array-like): ground truth target values
        y_pred (array-like): estimated target values
    
    Returns:
        float: root mean squared error value. lower is better
    """
    return math.sqrt(mean_squared_error(y_true, y_pred))
