from sklearn.model_selection import train_test_split


def normalize_pixels(X):
    """
    Scale pixel values from [0, 255] to [0, 1].
    Normalization helps maintain consistency across models
    and preprocessing pipelines.
    """
    return X / 255.0


def encode_labels(y):
    """
    Convert string labels returned by fetch_openml to integers.
    Example: ['0', '1', '9'] -> [0, 1, 9]
    """
    return y.astype(int)


def split_data(X, y):
    """
    Split the dataset into 80% training data and 20% test data
    using a fixed random seed for reproducibility.
    """
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )