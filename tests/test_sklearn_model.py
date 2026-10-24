import numpy as np
import pytest

from python import config, data, sklearn_model, weights

TOL = 1e-12


@pytest.fixture(scope="module")
def split():
    return data.load_csv()


@pytest.fixture
def loaded_model(split):
    X_train, _, y_train, _ = split
    clf = sklearn_model.build_model()
    sklearn_model.load_initial_into(clf, X_train, y_train)
    return clf


def test_n_train_matches_split(split):
    assert len(split[0]) == config.N_TRAIN


def test_learning_rate_comes_from_config(monkeypatch):
    assert sklearn_model.build_model().learning_rate_init == config.LEARNING_RATE
    monkeypatch.setattr(config, "LEARNING_RATE", 0.123)
    assert sklearn_model.build_model().learning_rate_init == 0.123


def test_weights_csv_roundtrip_is_exact():
    expected = weights.make_initial_weights()
    loaded = weights.load_initial_weights()
    for name in weights.NAMES:
        assert np.array_equal(expected[name], loaded[name]), name


def test_sklearn_forward_loss_matches_manual_and_reference(loaded_model, split):
    X_train, _, y_train, _ = split
    w = weights.load_initial_weights()
    manual = weights.manual_loss(w, X_train, y_train)
    assert abs(sklearn_model.cross_entropy(loaded_model, X_train, y_train) - manual) < TOL
    assert abs(weights.load_reference()["initial_train_loss"] - manual) < TOL


def test_first_epoch_loss_equals_initial_loss(loaded_model, split):
    X_train, _, y_train, _ = split
    loaded_model.partial_fit(X_train, y_train)
    reference = weights.load_reference()["initial_train_loss"]
    assert abs(loaded_model.loss_curve_[0] - reference) < TOL


def test_sklearn_matches_numpy_gradient_descent(loaded_model, split):
    """Independent numpy backprop: validates the in-place weight load and the GD equivalence."""
    X, _, y, _ = split
    w = weights.load_initial_weights()
    onehot = np.eye(config.N_CLASSES)[y]
    epochs = 20
    for _ in range(epochs):
        z1 = X @ w["w1"] + w["b1"]
        h = np.maximum(z1, 0.0)
        logits = h @ w["w2"] + w["b2"]
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        d2 = (p - onehot) / len(y)
        d1 = (d2 @ w["w2"].T) * (z1 > 0)
        grads = {"w2": h.T @ d2, "b2": d2.sum(0, keepdims=True),
                 "w1": X.T @ d1, "b1": d1.sum(0, keepdims=True)}
        for k in w:
            w[k] = w[k] - config.LEARNING_RATE * grads[k]
        loaded_model.partial_fit(X, y)
    assert np.allclose(loaded_model.coefs_[0], w["w1"], atol=1e-10, rtol=0)
    assert np.allclose(loaded_model.coefs_[1], w["w2"], atol=1e-10, rtol=0)
    assert np.allclose(loaded_model.intercepts_[0], w["b1"][0], atol=1e-10, rtol=0)
    assert np.allclose(loaded_model.intercepts_[1], w["b2"][0], atol=1e-10, rtol=0)


def test_exported_epoch_reference_matches_fresh_run(split):
    X_train, _, y_train, _ = split
    clf, _ = sklearn_model.train_partial_fit(X_train, y_train, epochs=config.CHECK_EPOCHS)
    ref = {n: np.loadtxt(config.DATA_DIR / f"ref_epoch_{n}.csv", delimiter=",", ndmin=2)
           for n in weights.NAMES}
    assert np.array_equal(clf.coefs_[0], ref["w1"])
    assert np.array_equal(clf.coefs_[1], ref["w2"])
    assert np.array_equal(clf.intercepts_[0], ref["b1"][0])
    assert np.array_equal(clf.intercepts_[1], ref["b2"][0])


def test_single_fit_follows_same_trajectory_as_partial_fit(split):
    X_train, _, y_train, _ = split
    by_partial, _ = sklearn_model.train_partial_fit(X_train, y_train)
    by_fit, _ = sklearn_model.train_single_fit(X_train, y_train)
    assert abs(sklearn_model.cross_entropy(by_partial, X_train, y_train)
               - sklearn_model.cross_entropy(by_fit, X_train, y_train)) < 1e-12
    assert np.allclose(by_partial.loss_curve_, by_fit.loss_curve_, atol=1e-12, rtol=0)
