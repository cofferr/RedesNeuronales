"""Shared initial weights and the reference loss used to validate both implementations.

Layout follows scikit-learn: W1 is (n_in, n_hidden), W2 is (n_hidden, n_out) and the
forward pass is ``relu(X @ W1 + b1) @ W2 + b2``. Matrices are written as header-less CSV
with 17 significant digits, so C++ reads exactly the same float64 values.
"""

import numpy as np
import pandas as pd

from . import config, data

NAMES = ("w1", "b1", "w2", "b2")
REFERENCE_CSV = config.DATA_DIR / "init_reference.csv"


def _path(name: str):
    return config.DATA_DIR / f"init_{name}.csv"


def make_initial_weights() -> dict[str, np.ndarray]:
    """He-normal for the ReLU layer, Glorot-normal for the output layer, zero biases."""
    rng = np.random.default_rng(config.SEED)
    n_in, n_h, n_out = config.N_FEATURES, config.N_HIDDEN, config.N_CLASSES
    return {
        "w1": rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_h)),
        "b1": np.zeros((1, n_h)),
        "w2": rng.normal(0.0, np.sqrt(2.0 / (n_h + n_out)), (n_h, n_out)),
        "b2": np.zeros((1, n_out)),
    }


def manual_forward(w: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    """Logits of the 4-8-3 network."""
    hidden = np.maximum(X @ w["w1"] + w["b1"], 0.0)
    return hidden @ w["w2"] + w["b2"]


def manual_loss(w: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> float:
    """Mean softmax cross-entropy, computed with a stable log-sum-exp."""
    logits = manual_forward(w, X)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(y)), y].mean())


def export_initial_weights() -> None:
    """Write the initial weights and the reference losses (computed in numpy) to data/."""
    config.DATA_DIR.mkdir(exist_ok=True)
    w = make_initial_weights()
    for name in NAMES:
        pd.DataFrame(w[name]).to_csv(
            _path(name), index=False, header=False, float_format=config.CSV_FLOAT_FORMAT
        )
    X_train, X_test, y_train, y_test = data.load_csv()
    reference = pd.DataFrame(
        {
            "name": ["initial_train_loss", "initial_test_loss"],
            "value": [manual_loss(w, X_train, y_train), manual_loss(w, X_test, y_test)],
        }
    )
    reference.to_csv(REFERENCE_CSV, index=False, float_format=config.CSV_FLOAT_FORMAT)


def load_initial_weights() -> dict[str, np.ndarray]:
    return {
        name: pd.read_csv(_path(name), header=None).to_numpy(dtype=np.float64)
        for name in NAMES
    }


def load_reference() -> dict[str, float]:
    df = pd.read_csv(REFERENCE_CSV)
    return dict(zip(df["name"], df["value"]))
