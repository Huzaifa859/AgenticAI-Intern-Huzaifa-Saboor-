import numpy as np
from classifier import normalize_pixels, encode_labels, split_data


# ── normalize_pixels tests ─────────────────────────────────────────────────────


def test_normalize_pixels_range():
    # All values should fall within [0, 1] after normalization
    X = np.array([[0, 128, 255], [64, 192, 32]], dtype=float)
    result = normalize_pixels(X)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_normalize_pixels_exact_values():
    # 0 should map to exactly 0.0 and 255 should map to exactly 1.0
    X = np.array([[0, 255]], dtype=float)
    result = normalize_pixels(X)
    assert result[0, 0] == 0.0
    assert result[0, 1] == 1.0


# ── encode_labels tests ────────────────────────────────────────────────────────


def test_encode_labels_dtype():
    # fetch_openml returns string labels — after encoding dtype must be int
    y = np.array(["0", "1", "2", "9"])
    result = encode_labels(y)
    assert result.dtype == int


def test_encode_labels_values():
    # String digits should map to their correct integer equivalents
    y = np.array(["0", "5", "9"])
    result = encode_labels(y)
    assert list(result) == [0, 5, 9]


# ── split_data tests ───────────────────────────────────────────────────────────


def test_split_data_shapes():
    # With test_size=0.2 on 100 samples: 80 train, 20 test
    X = np.zeros((100, 784))
    y = np.zeros(100)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20


def test_split_data_no_sample_loss():
    # Train + test should always add up to the original sample count
    X = np.zeros((100, 784))
    y = np.zeros(100)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert X_train.shape[0] + X_test.shape[0] == 100
