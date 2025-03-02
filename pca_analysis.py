import numpy as np
from sklearn.decomposition import PCA
from data_preprocessing import load_mnist

def compute_pca(n_components=10):
    """Computes PCA on the MNIST dataset."""
    (x_train, _), _ = load_mnist()
    pca = PCA(n_components=n_components).fit(x_train)
    return pca.components_

if __name__ == "__main__":
    components = compute_pca()
    print("PCA components shape:", components.shape)
