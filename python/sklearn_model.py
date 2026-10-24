"""Train and evaluate the scikit-learn MLPClassifier baseline.

Equivalence with the C++ implementation: plain full-batch gradient descent (sgd without
momentum, alpha=0, constant learning rate, batch_size = n_train, shuffle=False), float64,
and the same initial weights read from data/.

Two training modes:
- ``train_partial_fit``: one ``partial_fit`` call per epoch from the shared initial
  weights. Same trajectory as the C++ code, but timing includes Python per-call overhead.
- ``train_single_fit``: one ``fit`` call. This is the time comparable with C++.
"""

import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from . import config, data, weights


def build_model(max_iter: int = 1) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(config.N_HIDDEN,),
        activation="relu",
        solver="sgd",
        alpha=0.0,
        batch_size=config.N_TRAIN,
        learning_rate="constant",
        learning_rate_init=config.LEARNING_RATE,
        momentum=0.0,
        nesterovs_momentum=False,
        shuffle=False,
        early_stopping=False,
        max_iter=max_iter,
        tol=0.0,
        n_iter_no_change=config.EPOCHS + 1,
        random_state=config.SEED,
    )


def _classes() -> np.ndarray:
    return np.arange(config.N_CLASSES)


def load_initial_into(clf: MLPClassifier, X: np.ndarray, y: np.ndarray) -> None:
    """Prepare ``clf`` with the shared initial weights and an empty loss curve.

    The first ``partial_fit`` initializes the network (and takes one step, discarded here).
    Weights are overwritten IN PLACE: the SGD optimizer keeps references to the original
    arrays, so rebinding ``coefs_`` would silently train the wrong parameters.
    """
    clf.partial_fit(X, y, classes=_classes())
    w = weights.load_initial_weights()
    for layer, (wn, bn) in enumerate((("w1", "b1"), ("w2", "b2"))):
        clf.coefs_[layer][...] = w[wn]
        clf.intercepts_[layer][...] = w[bn][0]
    clf.loss_curve_ = []


def train_partial_fit(X, y, epochs: int = config.EPOCHS) -> tuple[MLPClassifier, float]:
    clf = build_model()
    load_initial_into(clf, X, y)
    start = time.perf_counter()
    for _ in range(epochs):
        clf.partial_fit(X, y)
    return clf, time.perf_counter() - start


def train_single_fit(X, y) -> tuple[MLPClassifier, float]:
    """One fit() call; it uses its own random init, but the cost per epoch is the same."""
    clf = build_model(max_iter=config.EPOCHS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        start = time.perf_counter()
        clf.fit(X, y)
        elapsed = time.perf_counter() - start
    assert clf.n_iter_ == config.EPOCHS
    return clf, elapsed


def cross_entropy(clf: MLPClassifier, X, y) -> float:
    proba = clf.predict_proba(X)
    return float(-np.log(proba[np.arange(len(y)), y]).mean())


def run() -> dict:
    """Train both modes, save the loss curve and a results JSON in results/."""
    import json

    import pandas as pd

    X_train, X_test, y_train, y_test = data.load_csv()
    clf, t_partial = train_partial_fit(X_train, y_train)
    _, t_single = train_single_fit(X_train, y_train)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"epoch": range(1, len(clf.loss_curve_) + 1), "loss": clf.loss_curve_}).to_csv(
        config.RESULTS_DIR / "sklearn_loss_curve.csv", index=False,
        float_format=config.CSV_FLOAT_FORMAT,
    )
    results = {
        "epochs": config.EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "train_loss": cross_entropy(clf, X_train, y_train),
        "test_loss": cross_entropy(clf, X_test, y_test),
        "train_accuracy": float(clf.score(X_train, y_train)),
        "test_accuracy": float(clf.score(X_test, y_test)),
        "time_single_fit_s": t_single,
        "time_partial_fit_loop_s": t_partial,
    }
    (config.RESULTS_DIR / "sklearn_results.json").write_text(json.dumps(results, indent=2))
    return results
