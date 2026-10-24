# Iris neural network: C++ (Eigen) vs Python (scikit-learn)

Final project for Redes Neuronales (Universidad Sergio Arboleda). The same
small neural network is implemented twice and compared:

- **C++17 + Eigen**: written from scratch (forward and backward propagation).
- **Python + scikit-learn**: `MLPClassifier`.

Dataset: Iris (150 samples, 4 features, 3 classes).
Architecture: 4 inputs -> 8 hidden neurons (ReLU) -> 3 outputs (softmax + cross-entropy), trained with gradient descent.
Both versions use identical hyperparameters and read the same train/test CSV files exported by Python to `data/`.

Comparison criteria: accuracy, loss, execution time, ease of implementation, pros and cons.

## Structure

```
.
├── pyproject.toml      Python project and dependencies (uv)
├── run_all.py          Runs the whole pipeline
├── python/             Data export, sklearn model, C++ runner, plots
├── cpp/                CMakeLists.txt + src/ (standalone executable, opens in Qt Creator)
├── data/               Train/test CSV files (versioned)
├── results/            Benchmark outputs and plots (not versioned)
└── README.md
```

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), CMake >= 3.16 and a C++17 compiler.
Eigen 3.4 is downloaded automatically by CMake if it is not installed.

```sh
uv sync
```

## Build and run

```sh
# C++ only
cmake -B cpp/build -S cpp && cmake --build cpp/build
cpp/build/nn_eigen

# Full pipeline (export data, sklearn, build + run C++, plots)
uv run run_all.py
```

To use Qt Creator, open `cpp/CMakeLists.txt`.

## Status

Project skeleton only; the pipeline stages in `run_all.py` are not implemented yet.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, `chore`, ...).
