"""Load Iris, split into train/test and export the splits to CSV files in data/."""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from . import config

TRAIN_CSV = config.DATA_DIR / "iris_train.csv"
TEST_CSV = config.DATA_DIR / "iris_test.csv"


def load_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split, standardized with the training statistics only."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED
    )
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std, y_train, y_test


def _to_frame(X: np.ndarray, y: np.ndarray):
    import pandas as pd  # only needed to export; keeps loading (and benchmarks) pandas-free

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["label"] = y
    return df


def export_csv() -> None:
    """Write the split to data/. Format: header, x0..x3 (float64), label (int)."""
    config.DATA_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()
    for path, X, y in ((TRAIN_CSV, X_train, y_train), (TEST_CSV, X_test, y_test)):
        _to_frame(X, y).to_csv(path, index=False, float_format=config.CSV_FLOAT_FORMAT)


def load_csv() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the exported CSV files (what both implementations train on).

    Plain numpy parsing (correctly rounded float64, like C++ std::stod) and no pandas, so
    the scikit-learn memory benchmark does not pay for importing it.
    """
    train = np.loadtxt(TRAIN_CSV, delimiter=",", skiprows=1)
    test = np.loadtxt(TEST_CSV, delimiter=",", skiprows=1)
    return train[:, :-1], test[:, :-1], train[:, -1].astype(int), test[:, -1].astype(int)
