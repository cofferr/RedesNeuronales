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

## Pipeline (`uv run run_all.py`)

1. Export the stratified, standardized Iris split, shared initial weights and reference values to `data/`.
2. Train scikit-learn (`partial_fit` per epoch for the equivalent trajectory; one `fit()` call, `warm_start` from the same weights, for timing).
3. Build the C++ project in Release and run it (`nn_eigen --train`).
4. Compare C++ vs scikit-learn: fails if losses differ by more than 1e-9 or accuracy/predictions differ.
5. Benchmarks (`--no-bench` skips them): interleaved timing blocks and clean-process memory runs, written to `results/benchmark.json`.
6. Figures (PNG + PDF) in `results/figures/` and the summary table in `results/summary_table.{csv,md}`.

`nn_eigen` also has `--gradcheck` (numerical gradient check) and runs forward/training checks against scikit-learn when called without arguments.
Tests: `uv run pytest`.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, `chore`, ...).
