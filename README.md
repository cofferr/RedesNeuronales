# Iris neural network: C++ (Eigen) vs Python (scikit-learn)

The same small neural network is implemented twice and compared:

- **C++17 + [Eigen](https://eigen.tuxfamily.org)**: forward and backward propagation written from scratch.
- **Python + [scikit-learn](https://scikit-learn.org)**: `MLPClassifier`.

Final project for *Redes Neuronales* (Universidad Sergio Arboleda).

| | |
|---|---|
| Dataset | Iris: 150 samples, 4 features, 3 classes (stratified 80/20 split, standardized) |
| Architecture | 4 inputs → 8 hidden neurons (ReLU) → 3 outputs (softmax + cross-entropy) |
| Training | Full-batch gradient descent; hyperparameters in [python/config.py](python/config.py) |

Both versions start from **the same data, the same initial weights and the same hyperparameters**, so they follow the same training trajectory. The pipeline checks this: it fails if the loss curves differ by more than `1e-9` or if the test predictions differ.

The comparison covers accuracy, loss, training time, memory, ease of implementation, and pros and cons. The written report is in [report/](report/) (LaTeX, in Spanish).

## Requirements

- [uv](https://docs.astral.sh/uv/) (installs Python 3.14 and the dependencies)
- CMake ≥ 3.16 and a C++17 compiler
- Internet access on the first build: CMake downloads Eigen 3.4 unless it is already installed
- Optional, to compile the report: [tectonic](https://tectonic-typesetting.github.io/) or a LaTeX distribution

## Quick start

```sh
uv sync                        # install Python dependencies
uv run run_all.py              # run the whole pipeline
uv run run_all.py --no-bench   # same, without the slower benchmarks
uv run pytest                  # run the tests
```

## What `uv run run_all.py` does

| # | Stage | Output |
|---|---|---|
| 1 | Export the split, shared initial weights and reference values | `data/` |
| 2 | Train scikit-learn (`partial_fit` per epoch for the trajectory, one `fit()` for timing) | `results/sklearn_*` |
| 3 | Build the C++ project in Release (`cmake`) | `cpp/build/nn_eigen` |
| 4 | Run the C++ binary (`nn_eigen --train`) | `results/cpp_*` |
| 5 | Compare C++ vs scikit-learn (loss curves, accuracy, predictions); stops on any mismatch | console |
| 6 | Benchmarks: interleaved timing blocks and clean-process memory runs | `results/benchmark.json` |
| 7 | Figures and summary table | `results/figures/`, `results/summary_table.{csv,md}` |
| 8 | Macros, tables and listings for the report, plus the Overleaf zip | `report/generated/`, `report_overleaf.zip` |

Python is the driver; the C++ program is compiled and launched by [python/cpp_runner.py](python/cpp_runner.py) and reads the CSV files that Python wrote to `data/`.
`--no-bench` skips stages 6 and 8.

## Running only the C++ part

```sh
cmake -B cpp/build -S cpp && cmake --build cpp/build
cpp/build/nn_eigen --train        # train and write results/cpp_*
cpp/build/nn_eigen --gradcheck    # numerical gradient check of the backward pass
cpp/build/nn_eigen                # forward/training checks against scikit-learn
cpp/build/noalloc_check           # checks that the training loop does not allocate
```

`data/` already contains the CSV files, so the C++ program runs without Python. To use Qt Creator, open `cpp/CMakeLists.txt`.

## Repository structure

```
.
├── run_all.py          Pipeline entry point
├── pyproject.toml      Python project and dependencies (uv); uv.lock pins versions
├── python/             Python side: data, sklearn model, C++ runner, benchmarks, plots
├── cpp/                C++ side: CMakeLists.txt and src/
├── data/               Train/test CSV files, initial weights, reference values (versioned)
├── results/            Pipeline outputs and figures (generated, not versioned)
├── tests/              pytest suite
└── report/             LaTeX report: main.tex, references.bib, figures/, generated/
```

## Important files

Start here to understand the project.

**C++** (`cpp/src/`)

| File | Purpose |
|---|---|
| [network.hpp](cpp/src/network.hpp) | The network: forward pass, backpropagation, gradient-descent step, preallocated buffers |
| [main.cpp](cpp/src/main.cpp) | Entry point: training loop, command-line modes, timing and memory measurement |
| [csv.hpp](cpp/src/csv.hpp) | Reads the CSV files |
| [gradcheck.hpp](cpp/src/gradcheck.hpp) | Compares analytic gradients with numerical ones |
| [noalloc_check.cpp](cpp/src/noalloc_check.cpp) | Checks that training does no heap allocation in Eigen |

**Python** (`python/`)

| File | Purpose |
|---|---|
| [config.py](python/config.py) | Hyperparameters and paths shared by both versions |
| [data.py](python/data.py) | Split, standardization and CSV export |
| [weights.py](python/weights.py) | Shared initial weights |
| [sklearn_model.py](python/sklearn_model.py) | The scikit-learn model and its reference trajectory |
| [compare.py](python/compare.py) | C++ vs scikit-learn equivalence check |
| [cpp_runner.py](python/cpp_runner.py) | Builds and runs the C++ binary from Python |
| [bench.py](python/bench.py), [bench_worker.py](python/bench_worker.py) | Time and memory benchmarks |
| [plots.py](python/plots.py) | Figures and summary table |
| [report_data.py](python/report_data.py) | Generates the report macros, tables and listings, and the Overleaf zip |

## Tests

`uv run pytest` covers the C++/scikit-learn equivalence, the no-allocation check, the data and config export, the benchmarks, the plots and the report generation.
The report test compiles `report/main.tex` and checks that it stays within ten pages; it is skipped if `tectonic` is not installed.

## Report

`report/` is self-contained. Every number in the text comes from `report/generated/numbers.tex`, produced by [python/report_data.py](python/report_data.py) from the pipeline results.

```sh
cd report && tectonic -X compile main.tex     # or: latexmk -pdf main.tex
uv run python -m python.report_data --zip     # report_overleaf.zip, to upload to Overleaf
```

On Overleaf, use the pdfLaTeX compiler with `main.tex` as the main file.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, `test`, `chore`, ...).
