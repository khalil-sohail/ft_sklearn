import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42):
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
    
    if (shuffle):
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(indices)

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test