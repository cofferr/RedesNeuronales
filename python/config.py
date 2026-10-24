"""Shared hyperparameters and paths, used identically by the Python and C++ versions."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Data split
SEED = 42
TEST_SIZE = 0.2
N_TRAIN = 120  # 150 samples * (1 - TEST_SIZE); full-batch size

# Architecture: 4 -> 8 (ReLU) -> 3 (softmax)
N_FEATURES = 4
N_HIDDEN = 8
N_CLASSES = 3

# Training (plain full-batch gradient descent)
LEARNING_RATE = 0.1  # final value
EPOCHS = 1000  # final value
CHECK_EPOCHS = 20  # short run used to validate the C++ backprop against sklearn

# CSV precision: 17 significant digits round-trips a float64 exactly
CSV_FLOAT_FORMAT = "%.17g"

CONFIG_CSV = DATA_DIR / "config.csv"


def export_config() -> None:
    """Write the values the C++ binary needs to data/config.csv (single source of truth)."""
    rows = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "check_epochs": CHECK_EPOCHS,
        "n_features": N_FEATURES,
        "n_hidden": N_HIDDEN,
        "n_classes": N_CLASSES,
        "n_train": N_TRAIN,
    }
    DATA_DIR.mkdir(exist_ok=True)
    lines = ["name,value"] + [f"{k},{v!r}" for k, v in rows.items()]
    CONFIG_CSV.write_text("\n".join(lines) + "\n")
