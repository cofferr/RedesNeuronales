import sys

import numpy as np
import pytest

from python import config, cpp_runner, data, sklearn_model


def test_cpp_and_sklearn_test_predictions_are_identical(tmp_path):
    try:
        binary = cpp_runner.binary_path()
    except cpp_runner.CppError:
        pytest.skip("C++ binary not built")
    import subprocess

    subprocess.run([str(binary), "--train", str(config.DATA_DIR), "--out", str(tmp_path)],
                   check=True, capture_output=True)
    cpp_pred = np.loadtxt(tmp_path / "cpp_predictions.csv", delimiter=",", skiprows=1, dtype=int)

    X_train, X_test, y_train, y_test = data.load_csv()
    clf, _ = sklearn_model.train_partial_fit(X_train, y_train)
    assert np.array_equal(cpp_pred[:, 0], y_test)
    assert np.array_equal(cpp_pred[:, 1], clf.predict(X_test))


def test_training_loop_does_not_allocate():
    """Preallocated buffers: 1000 epochs must run without any Eigen heap allocation."""
    import subprocess

    try:
        binary = cpp_runner.BUILD_DIR / "noalloc_check"
        assert binary.exists()
    except AssertionError:
        pytest.skip("noalloc_check not built")
    out = subprocess.run([str(binary)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "PASS" in out.stdout
