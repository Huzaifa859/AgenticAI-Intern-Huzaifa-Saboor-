from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns


# ── Utility functions ──────────────────────────────────────────────────────────


def normalize_pixels(X):
    """
    Scale pixel values from [0, 255] to [0, 1].
    Normalization helps gradient-based models converge faster.
    For Random Forest it has minimal impact but is good practice
    for pipeline consistency if models are swapped later.
    """
    return X / 255.0


def encode_labels(y):
    """
    Convert string labels returned by fetch_openml to integers.
    fetch_openml returns labels as strings by default ('0'–'9').
    scikit-learn classifiers expect numeric labels.
    """
    return y.astype(int)


def split_data(X, y):
    """
    Split dataset into 80% train and 20% test with a fixed random seed.
    random_state=42 ensures reproducibility across runs.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ── Pipeline ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load MNIST — 70,000 samples, 784 features (28x28 pixels), 10 classes
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data
    y = mnist.target

    # Preprocess
    X = normalize_pixels(X)
    y = encode_labels(y)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train — 100 trees, random_state for reproducibility
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    # Confusion matrix — annot=True shows counts in each cell
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", annot=True, fmt="d")
    plt.title("Confusion Matrix (MNIST — Random Forest)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()
