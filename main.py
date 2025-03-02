from data_preprocessing import load_mnist
from pca_analysis import compute_pca
from knn_faiss import knn_search
from model_training import train_model

def main():
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_mnist()

    print("Computing PCA...")
    compute_pca(10)

    print("Running kNN search...")
    knn_search(x_train, x_test[:10])

    print("Training models...")
    print("Logistic Regression Accuracy:", train_model("logistic"))
    print("SVM Accuracy:", train_model("svm"))

if __name__ == "__main__":
    main()
