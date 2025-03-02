import matplotlib.pyplot as plt
import seaborn as sns
from pca_analysis import compute_pca

def plot_pca_components():
    """Plots the first 10 PCA components."""
    components = compute_pca(10)
    fig, axes = plt.subplots(1, 10, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(components[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.show()

if __name__ == "__main__":
    plot_pca_components()
