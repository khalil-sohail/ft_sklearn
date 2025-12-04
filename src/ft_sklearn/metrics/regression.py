# MSE, MAE, RMSE
import numpy as np
import math

def mean_squared_error(y_true, y_pred):
    n_samples = len(y_true)
    s = sum(((y_true[i] - y_pred[i])**2) for i in range(0, n_samples))

    return (1/n_samples) * s

def mean_absolute_error(y_true, y_pred):
    n_samples = len(y_true)
    s = sum(abs(y_true[i] - y_pred[i]) for i in range(0, n_samples))

    return (1/n_samples) * s

def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))
