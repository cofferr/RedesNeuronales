import pandas as pd
import pytest

from python import compare, cpp_runner

SK = {"epochs": 1000, "learning_rate": 0.1, "train_loss": 0.05, "test_loss": 0.08,
      "train_accuracy": 0.975, "test_accuracy": 0.9667}
CURVE = pd.Series([1.0, 0.5, 0.25])


def test_identical_results_have_no_mismatches():
    assert compare.find_mismatches(SK, dict(SK), CURVE, CURVE.copy()) == []


def test_small_loss_difference_is_tolerated():
    cpp = {**SK, "train_loss": SK["train_loss"] + 1e-12}
    assert compare.find_mismatches(SK, cpp, CURVE, CURVE + 1e-12) == []


def test_large_loss_difference_fails():
    cpp = {**SK, "test_loss": SK["test_loss"] + 1e-6}
    assert any("test_loss" in p for p in compare.find_mismatches(SK, cpp, CURVE, CURVE))


def test_accuracy_must_be_identical():
    cpp = {**SK, "test_accuracy": SK["test_accuracy"] - 1e-9}
    assert any("test_accuracy" in p for p in compare.find_mismatches(SK, cpp, CURVE, CURVE))


def test_curve_difference_fails():
    assert any("loss curve" in p for p in compare.find_mismatches(SK, dict(SK), CURVE, CURVE + 1e-6))


def test_missing_cmake_gives_clear_error(monkeypatch):
    monkeypatch.setattr(cpp_runner.shutil, "which", lambda name: None)
    with pytest.raises(cpp_runner.CppError, match="cmake not found"):
        cpp_runner.check_toolchain()


def test_missing_compiler_gives_clear_error(monkeypatch):
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setattr(
        cpp_runner.shutil, "which", lambda name: "/usr/bin/cmake" if name == "cmake" else None
    )
    with pytest.raises(cpp_runner.CppError, match="no C\\+\\+ compiler"):
        cpp_runner.check_toolchain()
