import numpy as np

# --- Standard sklearn Imports ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- ft_sklearn Imports (Simulated for Comparison) ---
from ft_sklearn.model_selection import train_test_split as ft_train_test_split
from ft_sklearn.linear_model import LinearRegression as ft_LinearRegression
from ft_sklearn.metrics import mean_squared_error as ft_mean_squared_error


# 1. --- Generate Reproducible Random Data ---
np.random.seed(42) 
X = np.random.rand(100, 5) # 100 samples, 5 features
true_coefficients = np.array([2, 5, -1, 0.5, 3])
noise = 0.1 * np.random.randn(100)
y = X @ true_coefficients + 10 + noise # Target with intercept 10


## --- 2. Split Comparison ---
print("--- 🔬 Data Split Comparison ---")

# a) Standard sklearn split
X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=False
)

# b) ft_sklearn split (Simulated)
X_train_ft, X_test_ft, y_train_ft, y_test_ft = ft_train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=False
)

# Comparison Checks
x_train_equal = np.array_equal(X_train_sk, X_train_ft)
y_train_equal = np.array_equal(y_train_sk, y_train_ft)
x_test_equal = np.array_equal(X_test_sk, X_test_ft)
y_test_equal = np.array_equal(y_test_sk, y_test_ft)

print(f"sk X_train Shape: {X_train_sk.shape}")
print(f"ft X_train Shape: {X_train_ft.shape}")

print(f"\nX_train Equal? **{x_train_equal}**")
print(f"y_train Equal? **{y_train_equal}**")
print(f"X_test Equal? **{x_test_equal}**")
print(f"y_test Equal? **{y_test_equal}**")

# Use the sk split for the rest of the comparison since they are identical
X_train, X_test, y_train, y_test = X_train_sk, X_test_sk, y_train_sk, y_test_sk


# --- 3. Standard sklearn Linear Regression ---
print("\n--- ⚙️ Standard sklearn Regression Results ---")

model_sk = LinearRegression().fit(X_train, y_train)
y_pred_sk = model_sk.predict(X_test)
mse_sk = mean_squared_error(y_test, y_pred_sk)

print(f"  Coefficients (sk_coef_): {model_sk.coef_}")
print(f"  Intercept (sk_intercept_): {model_sk.intercept_:.4f}")
print(f"  Mean Squared Error (sk_mse): {mse_sk:.6f}")


# --- 4. ft_sklearn Linear Regression (Simulated) ---
# REPLACE THESE CALLS with your actual ft_sklearn library functions
print("\n--- 🧪 ft_sklearn Simulated Regression Results ---")

ft_model = ft_LinearRegression().fit(X_train, y_train)
ft_y_pred = ft_model.predict(X_test)
mse_ft = ft_mean_squared_error(y_test, ft_y_pred)

print(f"  Coefficients (ft_coef_): {ft_model.coef_}")
print(f"  Intercept (ft_intercept_): {ft_model.intercept_:.4f}")
print(f"  Mean Squared Error (ft_mse): {mse_ft:.6f}")


# --- 5. Comparison Summary ---
print("\n--- 🤝 Final Comparison Summary ---")
print(f"Difference in MSE: {abs(mse_sk - mse_ft):.10f}")
print("---")