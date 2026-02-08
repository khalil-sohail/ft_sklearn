from sklearn.preprocessing import RobustScaler
from ft_sklearn.preprocessing import RobustScaler as ft_RobustScaler
import numpy as np


# Helper: Compare sklearn vs ft results (shared logic)
def compare_scalers(X, sklearn_scaler, ft_scaler):
    X_sklearn = sklearn_scaler.fit_transform(X)
    ft_X = ft_scaler.fit_transform(X)

    print("Comparing Results…")
    print("-" * 40)

    # 1. Check Shape
    if not np.array_equal(X_sklearn.shape, ft_X.shape):
        print("❌ Shape mismatch.")
        print("Sklearn:", X_sklearn.shape)
        print("ft:", ft_X.shape)
        return

    # 2. Check Output Values
    if np.allclose(X_sklearn, ft_X, atol=1e-7):
        print("✅ Transformed data matches.")
    else:
        print("❌ Transformed data mismatch.")
        # (Optional: Print specific mismatches here if needed)

    # 3. Check Attributes (Safely)
    checks = [
        ("center_", sklearn_scaler.center_, ft_scaler.center_),
        ("scale_", sklearn_scaler.scale_, ft_scaler.scale_),
        ("quantile_range", sklearn_scaler.quantile_range, ft_scaler.quantile_range),
    ]

    for name, sk, ft in checks:
        # CASE A: Both are None (Correct behavior for disabled options)
        if sk is None and ft is None:
            print(f"✅ {name} match (Both None).")
            continue

        # CASE B: One is None and the other isn't (Mismatch)
        if sk is None or ft is None:
            print(f"❌ {name} mismatch. Sk: {type(sk)}, Ft: {type(ft)}")
            continue

        # CASE C: Both are arrays/tuples (Do the math)
        try:
            if np.allclose(sk, ft, atol=1e-9):
                print(f"✅ {name} match.")
            else:
                print(f"❌ {name} mismatch.")
                print("Sk:", sk)
                print("ft:", ft)
        except Exception as e:
            print(f"❌ Error comparing {name}: {e}")

    # 4. Check Inverse Transform
    print("\nTesting inverse_transform()...")
    
    # Handle inverse transform even if scaling was disabled
    try:
        X_sklearn_inv = sklearn_scaler.inverse_transform(X_sklearn)
        ft_X_inv = ft_scaler.inverse_transform(ft_X)

        if np.allclose(X_sklearn_inv, ft_X_inv, atol=1e-7):
            print("✅ inverse_transform outputs match.")
        else:
            print("❌ inverse_transform mismatch.")
            
        # Round-trip correctness
        if np.allclose(ft_X_inv, X, atol=1e-7):
             print("✅ ft round-trip OK (Original -> Scaled -> Original).")
        else:
             print("❌ ft round-trip FAILED.")
             
    except Exception as e:
        print(f"❌ Crash during inverse_transform: {e}")

    print("\n")


# Test 1: Default RobustScaler
def test_robust_default():
    print("🧪 Test 1 — Default RobustScaler (copy=True)\n")

    X = np.array([
        [1.0, 100.0, -10.0],
        [2.0, 150.0, -5.0],
        [3.0, 200.0, 0.0],
        [4.0, 250.0, 5.0],
        [5.0, 300.0, 10.0]
    ])

    sklearn_scaler = RobustScaler(copy=True)
    ft_scaler = ft_RobustScaler(copy=True)

    compare_scalers(X, sklearn_scaler, ft_scaler)


# Test 2: RobustScaler with copy=False
def test_robust_copy_false():
    print("🧪 Test 2 — RobustScaler with copy=False\n")

    X = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
        [30.0, 300.0]
    ])

    X_sk = X.copy()
    X_ft = X.copy()

    sklearn_scaler = RobustScaler(copy=False)
    ft_scaler = ft_RobustScaler(copy=False)

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


# Test 3: RobustScaler without centering or scaling
def test_robust_disable_options():
    print("🧪 Test 3 — RobustScaler with centering/scaling toggled\n")

    X = np.array([
        [1.0, 100.0],
        [2.0, 150.0],
        [3.0, 200.0]
    ])

    configs = [
        (True, False),
        (False, True),
        (False, False)
    ]

    for with_centering, with_scaling in configs:
        print(f"Testing with_centering={with_centering}, with_scaling={with_scaling}")

        sklearn_scaler = RobustScaler(
            with_centering=with_centering,
            with_scaling=with_scaling
        )
        ft_scaler = ft_RobustScaler(
            with_centering=with_centering,
            with_scaling=with_scaling
        )

        compare_scalers(X, sklearn_scaler, ft_scaler)


# Run tests
if __name__ == "__main__":
    test_robust_default()
    test_robust_copy_false()
    test_robust_disable_options()
