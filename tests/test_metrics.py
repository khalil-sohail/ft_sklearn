from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,

    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
)

from ft_sklearn.metrics import (
    accuracy_score as ft_accuracy_score,
    precision_score as ft_precision_score,
    recall_score as ft_recall_score,
    f1_score as ft_f1_score,

    mean_squared_error as ft_mean_squared_error,
    mean_absolute_error as ft_mean_absolute_error,
    root_mean_squared_error as ft_root_mean_squared_error,
)
import numpy as np

# Classification example
y_true_class = [0, 1, 1, 0, 1, 0]
y_pred_class = [0, 1, 0, 0, 1, 1]

# Regression example
y_true_reg = [3.0, 5.0, 2.5, 7.0]
y_pred_reg = [2.5, 5.0, 4.0, 8.0]

# Classification metrics
accuracy = accuracy_score(y_true_class, y_pred_class)
ft_accuracy = ft_accuracy_score(y_true_class, y_pred_class)

recall = recall_score(y_true_class, y_pred_class)
ft_recall = ft_recall_score(y_true_class, y_pred_class)

precision = precision_score(y_true_class, y_pred_class)
ft_precision = ft_precision_score(y_true_class, y_pred_class)

f1 = f1_score(y_true_class, y_pred_class)
ft_f1 = ft_f1_score(y_true_class, y_pred_class)

print("Classification Metrics:")
print(f"Accuracy  test: {accuracy == ft_accuracy}")
print(f"Recall    test: {recall == ft_recall}")
print(f"Precision test: {precision == ft_precision}")
print(f"f1        test: {f1 == ft_f1}")

# Regression metrics
mse = mean_squared_error(y_true_reg, y_pred_reg)
ft_mse = ft_mean_squared_error(y_true_reg, y_pred_reg)

mae = mean_absolute_error(y_true_reg, y_pred_reg)
ft_mae = ft_mean_absolute_error(y_true_reg, y_pred_reg)

rmse = root_mean_squared_error(y_true_reg, y_pred_reg)
ft_rmse = ft_root_mean_squared_error(y_true_reg, y_pred_reg)

print("\nRegression Metrics:")
print(f"MSE  test: {mse == ft_mse}")
print(f"MAE  test: {mae == ft_mae}")
print(f"RMSE test: {rmse == ft_rmse}")
