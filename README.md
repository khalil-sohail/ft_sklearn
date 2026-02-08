# ft_sklearn

A lightweight machine learning library built from scratch to implement core scikit-learn functionality. This project demonstrates fundamental ML concepts including regression, preprocessing, metrics, and model selection.

## Overview

ft_sklearn provides a collection of machine learning tools organized into intuitive modules. All implementations follow scikit-learn's API conventions for consistency and ease of use.

## Installation

Install the package using pip:

```bash
pip install -e /path/to/ft_sklearn
```

Or directly:

```bash
cd ft_sklearn
pip install -e .
```

## Requirements

- Python 3.11 or higher
- NumPy

## Project Structure

### Linear Models (`linear_model/`)

- **LinearRegression**: Implements ordinary least squares regression using matrix operations
- **SGDRegressor**: Stochastic gradient descent regressor (framework provided for implementation)
- **LogisticRegression**: Logistic Regression classifier

### Preprocessing (`preprocessing/`)

Feature scaling and transformation utilities:

- **StandardScaler**: Standardizes features by removing the mean and scaling to unit variance
- **MinMaxScaler**: Scales features to a specified range, typically [0, 1]
- **RobustScaler**: Scales features using statistics robust to outliers (median and interquartile range)

### Classification

- **DecisionTreeClassifier**: Decision Tree classifier
- **SVC**: Support Vector Classification
- **GaussianNB**: Gaussian Naive Bayes
- **KNeighborsClassifier**: K-Nearest Neighbors classifier

### Clustering

- **KMeans**: K-Means clustering

### Ensemble

- **RandomForestClassifier**: Random Forest classifier
- **GradientBoostingClassifier**: Gradient Boosting classifier

### Decomposition

- **PCA**: Principal Component Analysis

### Metrics (`metrics/`)

Model evaluation functions:

- **Regression Metrics** (`regression.py`):
  - `mean_squared_error()`: MSE - average of squared errors
  - `mean_absolute_error()`: MAE - average of absolute errors
  - `root_mean_squared_error()`: RMSE - square root of MSE

- **Classification Metrics** (`classification.py`):
  - `accuracy_score()`: Proportion of correct predictions
  - `precision_score()`: Proportion of positive predictions that were correct
  - `recall_score()`: Proportion of actual positives correctly identified
  - `f1_score()`: Harmonic mean of precision and recall

### Model Selection (`model_selection/`)

- **train_test_split()**: Splits data into training and testing sets with optional shuffling

### Utilities (`utils/`)

- **type_check.py**: Data validation functions
  - `check_X_y()`: Validates feature matrix and target vector
  - `check_X()`: Validates feature matrix

### Base Classes (`base.py`)

- **BaseEstimator**: Base class providing parameter management via `get_params()` and `set_params()`
- **ClassifierMixin**: Mixin for classifier models
- **RegressorMixin**: Mixin for regression models

## Usage Examples

### Regression

```python
from ft_sklearn.linear_model import LinearRegression
from ft_sklearn.model_selection import train_test_split
from ft_sklearn.metrics import mean_squared_error
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([2, 4, 5, 4])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")
```

### Feature Scaling

```python
from ft_sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# scale new data with the same transformation
X_new = np.array([[5, 6]])
X_new_scaled = scaler.transform(X_new)

# reverse the transformation
X_original = scaler.inverse_transform(X_scaled)
```

### Classification Metrics

```python
from ft_sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
```

## API Conventions

All estimators follow scikit-learn conventions:

- `fit()`: Learn parameters from training data
- `predict()`: Make predictions on new data
- `transform()`: Apply learned transformation (preprocessing)
- `fit_transform()`: Fit and transform in one step
- `inverse_transform()`: Reverse a transformation (preprocessing)
- `get_params()`: Get estimator parameters
- `set_params()`: Set estimator parameters

## Testing

Run the test suite:

```bash
pytest test/
```

Tests are provided for:
- LinearRegression
- Data preprocessing (StandardScaler, MinMaxScaler, RobustScaler)
- Metrics (classification and regression)
- Train-test split functionality

## License

Educational project for learning ML fundamentals.
