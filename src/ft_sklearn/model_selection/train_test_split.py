"""model selection utilities for data splitting

provides tools for splitting data into training and testing sets
"""

import numpy as np


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42):
    """split arrays or matrices into random train and test subsets
    
    this is a utility for dividing datasets into training and testing portions.
    by default, the data is shuffled before splitting to ensure randomness
    
    Args:
        X (array-like): feature matrix of shape (n_samples, n_features)
        y (array-like): target values of shape (n_samples,)
        test_size (float or int, optional): if float, represents the proportion
            of the dataset to include in the test split (between 0 and 1).
            if int, represents the absolute number of test samples.
            default is 0.2 (20%)
        shuffle (bool, optional): if True, shuffle data before splitting.
            default is True
        random_state (int, optional): random seed for reproducibility.
            default is 42
    
    Returns:
        tuple: four arrays (X_train, X_test, y_train, y_test) where:
            X_train: training feature matrix
            X_test: testing feature matrix
            y_train: training target values
            y_test: testing target values
    
    Raises:
        ValueError: if test_size is not a float or integer
    
    Example:
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2, 3, 4])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    """
    num_samples = len(X)

    if isinstance(test_size, float):
        n_test = int(num_samples * test_size)
    elif isinstance(test_size, int):
        n_test = test_size
    else:
        raise ValueError("test_size must be a float or an integer.")

    n_train = num_samples - n_test
    split_index = n_train

    indices = np.arange(num_samples)

    if shuffle:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(indices)

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test