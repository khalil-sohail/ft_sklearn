import numpy as np
import pytest
from ft_sklearn.decomposition.PCA import PCA

@pytest.mark.skip(reason="Not implemented yet")
def test_pca_transformation():
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4]
    ])
    
    pca = PCA(n_components=1)
    X_transformed = pca.fit_transform(X)
    
    assert X_transformed.shape == (4, 1)
    
    X_reconstructed = pca.inverse_transform(X_transformed)
    assert np.allclose(X, X_reconstructed)