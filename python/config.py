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
LEARNING_RATE = 0.1
EPOCHS = 1000

# CSV precision: 17 significant digits round-trips a float64 exactly
CSV_FLOAT_FORMAT = "%.17g"
