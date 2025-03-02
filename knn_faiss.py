import faiss
import numpy as np
from data_preprocessing import load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist()

def knn_search(x_train, x_test, k=1):
    """Performs nearest neighbor search using Faiss."""
    index = faiss.IndexFlatL2(x_train.shape[1])
    index.add(x_train)
    _, idx = index.search(x_test, k)
    return idx

if __name__ == "__main__":
    result = knn_search(x_train, x_test[:10])
    print("Nearest neighbors:", result)
