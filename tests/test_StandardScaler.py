from sklearn.preprocessing import StandardScaler
from ft_sklearn.preprocessing import StandardScaler as ft_StandardScaler
import numpy as np

def test_scaler():
    print("🧪 Starting StandardScaler Tests...\n")

    # A. Synthetic data
    X_orig = np.array([
        [1.0, 200.0, -10.0],
        [2.0, 250.0, -5.0],
        [3.0, 200.0, 0.0],
        [4.0, 300.0, 5.0],
        [5.0, 250.0, 10.0]
    ])
    print(f"Original Data X (Shape {X_orig.shape}):\n{X_orig}")
    print("-" * 40)

    # --- Test 1: Default copy=True, with_mean=True, with_std=True ---
    print("Test 1 — Default StandardScaler")
    X = X_orig.copy()
    sklearn_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_sklearn = sklearn_scaler.fit_transform(X)

    ft_scaler = ft_StandardScaler(copy=True, with_mean=True, with_std=True)
    X_ft = ft_scaler.fit_transform(X_orig.copy())

    compare_results(X_sklearn, X_ft, sklearn_scaler, ft_scaler, X_orig)

    # --- Test 2: copy=False ---
    print("\nTest 2 — copy=False")
    X = X_orig.copy()
    X_ft = X_orig.copy()
    sklearn_scaler = StandardScaler(copy=False)
    ft_scaler = ft_StandardScaler(copy=False)
    X_sklearn = sklearn_scaler.fit_transform(X)
    X_ft_trans = ft_scaler.fit_transform(X_ft)

    print("Sklearn X mutated:\n", X)
    print("ft X mutated:\n", X_ft)
    if np.allclose(X_sklearn, X_ft_trans, atol=1e-7):
        print("✅ Transformed outputs match.")
    else:
        print("❌ Transformed outputs mismatch.")

    # --- Test 3: with_mean=False ---
    print("\nTest 3 — with_mean=False")
    X = X_orig.copy()
    sklearn_scaler = StandardScaler(with_mean=False)
    ft_scaler = ft_StandardScaler(with_mean=False)
    X_sklearn = sklearn_scaler.fit_transform(X)
    X_ft = ft_scaler.fit_transform(X_orig.copy())

    compare_results(X_sklearn, X_ft, sklearn_scaler, ft_scaler, X_orig)

    # --- Test 4: with_std=False ---
    print("\nTest 4 — with_std=False")
    X = X_orig.copy()
    sklearn_scaler = StandardScaler(with_std=False)
    ft_scaler = ft_StandardScaler(with_std=False)
    X_sklearn = sklearn_scaler.fit_transform(X)
    X_ft = ft_scaler.fit_transform(X_orig.copy())

    compare_results(X_sklearn, X_ft, sklearn_scaler, ft_scaler, X_orig)


def compare_results(X_sklearn, X_ft, sklearn_scaler, ft_scaler, X_orig):
    print("-" * 40)
    # Compare shapes
    if X_sklearn.shape != X_ft.shape:
        print("❌ Shape mismatch.")
        return

    # Compare scaled output
    if np.allclose(X_sklearn, X_ft, atol=1e-7):
        print("✅ Transformed data matches.")
    else:
        print("❌ Transformed data mismatch.")
        print("Sklearn:\n", X_sklearn)
        print("ft:\n", X_ft)

    # Compare mean
    if sklearn_scaler.mean_ is not None and ft_scaler.mean_ is not None:
        if np.allclose(sklearn_scaler.mean_, ft_scaler.mean_, atol=1e-9):
            print("✅ Means match.")
        else:
            print("❌ Mean mismatch.")
            print("Sk:", sklearn_scaler.mean_)
            print("ft:", ft_scaler.mean_)

    # Compare scale
    if sklearn_scaler.scale_ is not None and ft_scaler.scale_ is not None:
        if np.allclose(sklearn_scaler.scale_, ft_scaler.scale_, atol=1e-9):
            print("✅ Scales match.")
        else:
            print("❌ Scale mismatch.")
            print("Sk:", sklearn_scaler.scale_)
            print("ft:", ft_scaler.scale_)

    # Test inverse_transform
    X_sklearn_inv = sklearn_scaler.inverse_transform(X_sklearn)
    X_ft_inv = ft_scaler.inverse_transform(X_ft)
    if np.allclose(X_sklearn_inv, X_ft_inv, atol=1e-7):
        print("✅ inverse_transform outputs match.")
    else:
        print("❌ inverse_transform mismatch.")

    if np.allclose(X_sklearn_inv, X_orig, atol=1e-7):
        print("✅ sklearn inverse_transform restores X.")
    else:
        print("❌ sklearn inverse_transform does NOT restore X.")

    if np.allclose(X_ft_inv, X_orig, atol=1e-7):
        print("✅ ft inverse_transform restores X.")
    else:
        print("❌ ft inverse_transform does NOT restore X.")


if __name__ == "__main__":
    test_scaler()
