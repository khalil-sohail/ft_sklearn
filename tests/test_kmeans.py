import numpy as np
import pytest
from ft_sklearn.cluster.KMeans import KMeans

@pytest.mark.skip(reason="Not implemented yet")
def test_kmeans_clustering():
    X = np.array([
        [1, 1], [1, 2], [2, 1],
        [10, 10], [10, 11], [11, 10]
    ])
    
    model = KMeans(n_clusters=2)
    model.fit(X)
    
    labels = model.predict(X)
    
    assert len(np.unique(labels)) == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]