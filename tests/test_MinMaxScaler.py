from sklearn.preprocessing import MinMaxScaler
from ft_sklearn.preprocessing import MinMaxScaler as ft_MinMaxScaler
import numpy as np


# ------------------------------------------------------------
# Helper: Compare sklearn vs ft results (shared logic)
# ------------------------------------------------------------
def compare_scalers(X, sklearn_scaler, ft_scaler):
    X_sklearn = sklearn_scaler.fit_transform(X)
    ft_X = ft_scaler.fit_transform(X)

    print("Comparing Results…")
    print("-" * 40)

    # --- Compare shapes ---
    if X_sklearn.shape != ft_X.shape:
        print("❌ Shape mismatch.")
        print("Sklearn shape:", X_sklearn.shape)
        print("ft shape:", ft_X.shape)
        return

    # --- Compare scaled output ---
    if np.allclose(X_sklearn, ft_X, atol=1e-7, rtol=1e-7):
        print("✅ Transformed data matches.")
    else:
        print("❌ Transformed data mismatch.")
        for i in range(X.shape[1]):
            if not np.allclose(X_sklearn[:, i], ft_X[:, i], atol=1e-7, rtol=1e-7):
                print(f"Feature {i} differs:")
                print("Sk:", X_sklearn[:, i])
                print("ft_sk:", ft_X[:, i])

    # --- Compare attributes ---
    checks = [
        ("data_min_", sklearn_scaler.data_min_, ft_scaler.data_min_),
        ("data_max_", sklearn_scaler.data_max_, ft_scaler.data_max_),
        ("scale_", sklearn_scaler.scale_, ft_scaler.scale_),
        ("min_", sklearn_scaler.min_, ft_scaler.min_),
    ]

    for name, sk, ft in checks:
        if np.allclose(sk, ft, atol=1e-9):
            print(f"✅ {name} match.")
        else:
            print(f"❌ {name} mismatch.")
            print("Sk:", sk)
            print("ft_sk:", ft)

    # --------------------------------------------------
    # inverse_transform test
    # --------------------------------------------------
    print("\nTesting inverse_transform()...")
    print("-" * 40)

    X_sklearn_inv = sklearn_scaler.inverse_transform(X_sklearn)
    ft_X_inv = ft_scaler.inverse_transform(ft_X)

    if np.allclose(X_sklearn_inv, ft_X_inv, atol=1e-7, rtol=1e-7):
        print("✅ inverse_transform outputs match.")
    else:
        print("❌ inverse_transform mismatch.")
        print("Sk:", X_sklearn_inv)
        print("ft_sk:", ft_X_inv)

    # round-trip verify
    if np.allclose(X_sklearn_inv, X, atol=1e-7):
        print("✅ sklearn inverse_transform restores X.")
    else:
        print("❌ sklearn round-trip failed.")

    if np.allclose(ft_X_inv, X, atol=1e-7):
        print("✅ ft inverse_transform restores X.")
    else:
        print("❌ ft round-trip failed.")

    print("\n")


# ------------------------------------------------------------
# Test 1: Default MinMaxScaler
# ------------------------------------------------------------
def test_minmax_default():
    print("🧪 Test 1 — Default MinMaxScaler (copy=True, clip=False)\n")

    X = np.array([
        [1.0, 200.0, -10.0],
        [2.0, 250.0, -5.0],
        [3.0, 200.0, 0.0],
        [4.0, 300.0, 5.0],
        [5.0, 250.0, 10.0]
    ])

    sklearn_scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    ft_scaler = ft_MinMaxScaler(feature_range=(0, 1), copy=True)

    compare_scalers(X, sklearn_scaler, ft_scaler)


# ------------------------------------------------------------
# Test 2: MinMaxScaler with clip=True
# ------------------------------------------------------------
def test_minmax_clip():
    print("🧪 Test 2 — MinMaxScaler with clip=True\n")

    X = np.array([
        [0.0, 100.0],
        [10.0, 200.0],
        [20.0, 300.0]
    ])

    # force out-of-range value to confirm clipping works
    X_test = np.array([[25.0, 50.0]])

    sklearn_scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    ft_scaler = ft_MinMaxScaler(feature_range=(0, 1), clip=True)

    sklearn_scaler.fit(X)
    ft_scaler.fit(X)

    sk_t = sklearn_scaler.transform(X_test)
    ft_t = ft_scaler.transform(X_test)

    print("Sk:", sk_t)
    print("ft:", ft_t)

    if np.allclose(sk_t, ft_t, atol=1e-7):
        print("✅ clip=True behavior matches sklearn.\n")
    else:
        print("❌ clip=True mismatch.\n")


# ------------------------------------------------------------
# Test 3: MinMaxScaler copy=False
# ------------------------------------------------------------
def test_minmax_copy_false():
    print("🧪 Test 3 — MinMaxScaler with copy=False\n")

    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0]
    ])

    X_ft = X.copy()
    X_sk = X.copy()

    sklearn_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    ft_scaler = ft_MinMaxScaler(feature_range=(0, 1), copy=False)

    sk_out = sklearn_scaler.fit_transform(X_sk)
    ft_out = ft_scaler.fit_transform(X_ft)

    print("Sk output:", sk_out)
    print("ft output:", ft_out)

    if np.allclose(sk_out, ft_out, atol=1e-7):
        print("✅ copy=False transformed output matches.")
    else:
        print("❌ copy=False transformed output mismatch.")

    print("Sk X mutated:", X_sk)
    print("ft X mutated:", X_ft)

    if np.allclose(X_sk, X_ft, atol=1e-7):
        print("✅ copy=False mutation behavior matches sklearn.\n")
    else:
        print("❌ copy=False mutation behavior differs.\n")


# ------------------------------------------------------------
# Run all tests
# ------------------------------------------------------------
if __name__ == "__main__":
    test_minmax_default()
    test_minmax_clip()
    test_minmax_copy_false()
