from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data_preprocessing import load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist()

def train_model(model_type="logistic"):
    """Trains a classification model (Logistic Regression or SVM)."""
    if model_type == "logistic":
        model = LogisticRegression(max_iter=10000)
    elif model_type == "svm":
        model = SVC(kernel='linear')

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

if __name__ == "__main__":
    log_accuracy = train_model("logistic")
    print("Logistic Regression Accuracy:", log_accuracy)

    svm_accuracy = train_model("svm")
    print("SVM Accuracy:", svm_accuracy)
