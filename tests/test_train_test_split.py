# from ft_sklearn.model_selection import train_test_split
from ft_sklearn.model_selection import train_test_split
import numpy as np


def test_basic_split():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    total = len(X_train) + len(X_test)
    assert total == len(X)

    print("shuffle behavior OK ✓")

def test_no_shuffle():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    assert np.array_equal(X_train, X[:8])
    assert np.array_equal(X_test, X[8:])
    assert np.array_equal(y_train, y[:8])
    assert np.array_equal(y_test, y[8:])
    print("No-shuffle behavior OK ✓")

def test_edge_cases():
    X = np.arange(10)
    y = np.arange(10)

    # test_size as int
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3)
    assert len(X_test) == 3
    assert len(X_train) == 7

    # test_size as float
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.5)
    assert len(X_test) == 5
    assert len(X_train) == 5

    print("Edge cases OK ✓")

if __name__ == "__main__":
    test_basic_split()
    test_no_shuffle()
    test_edge_cases()
