"""Compare the C++ and scikit-learn results; fail loudly if they are not equivalent."""

import pandas as pd

from . import config

LOSS_TOL = 1e-9
LOSS_KEYS = ("train_loss", "test_loss")
ACCURACY_KEYS = ("train_accuracy", "test_accuracy")


def load_curves() -> tuple[pd.Series, pd.Series]:
    sk = pd.read_csv(config.RESULTS_DIR / "sklearn_loss_curve.csv")["loss"]
    cpp = pd.read_csv(config.RESULTS_DIR / "cpp_loss_curve.csv")["loss"]
    return sk, cpp


def find_mismatches(sk: dict, cpp: dict, sk_curve, cpp_curve) -> list[str]:
    """Return a human readable list of everything that differs beyond the tolerances."""
    problems = []
    for key in ("epochs", "learning_rate"):
        if sk[key] != cpp[key]:
            problems.append(f"{key}: sklearn {sk[key]} vs C++ {cpp[key]}")
    for key in LOSS_KEYS:
        diff = abs(sk[key] - cpp[key])
        if diff > LOSS_TOL:
            problems.append(f"{key}: |diff| = {diff:.3e} > {LOSS_TOL:.0e}")
    for key in ACCURACY_KEYS:
        if sk[key] != cpp[key]:
            problems.append(f"{key}: sklearn {sk[key]} vs C++ {cpp[key]} (must be identical)")
    if len(sk_curve) != len(cpp_curve):
        problems.append(f"loss curve length: sklearn {len(sk_curve)} vs C++ {len(cpp_curve)}")
    else:
        worst = float((sk_curve - cpp_curve).abs().max())
        if worst > LOSS_TOL:
            problems.append(f"loss curve: max |diff| = {worst:.3e} > {LOSS_TOL:.0e}")
    return problems


def summary(sk: dict, cpp: dict, sk_curve, cpp_curve) -> str:
    rows = [
        ("train loss", sk["train_loss"], cpp["train_loss"]),
        ("test loss", sk["test_loss"], cpp["test_loss"]),
        ("train accuracy", sk["train_accuracy"], cpp["train_accuracy"]),
        ("test accuracy", sk["test_accuracy"], cpp["test_accuracy"]),
        ("train time (s)", sk["time_train_s"], cpp["time_train_s"]),
    ]
    lines = [f"{'metric':<16}{'sklearn':>14}{'C++':>14}"]
    lines += [f"{name:<16}{a:>14.6g}{b:>14.6g}" for name, a, b in rows]
    lines.append(f"sklearn partial_fit loop (Python overhead): {sk['time_partial_fit_loop_s']:.4f} s")
    lines.append(f"speedup C++ vs sklearn fit(): {sk['time_train_s'] / cpp['time_train_s']:.1f}x")
    if len(sk_curve) == len(cpp_curve):
        lines.append(f"loss curve max |diff|: {float((sk_curve - cpp_curve).abs().max()):.3e}")
    return "\n".join(lines)


def prediction_mismatches() -> list[str]:
    """Test-set predictions of both implementations must be identical."""
    sk = pd.read_csv(config.RESULTS_DIR / "sklearn_predictions.csv")
    cpp = pd.read_csv(config.RESULTS_DIR / "cpp_predictions.csv")
    if not sk["true"].equals(cpp["true"]):
        return ["test labels differ between sklearn and C++ prediction files"]
    differing = int((sk["pred"] != cpp["pred"]).sum())
    return [f"{differing} test predictions differ between sklearn and C++"] if differing else []
