"""Build the C++ project with CMake, run the binary and collect its results."""

import json
import os
import shutil
import subprocess

from . import config

CPP_DIR = config.ROOT / "cpp"
BUILD_DIR = CPP_DIR / "build"
CPP_RESULTS_JSON = config.RESULTS_DIR / "cpp_results.json"
CPP_LOSS_CURVE_CSV = config.RESULTS_DIR / "cpp_loss_curve.csv"


class CppError(RuntimeError):
    """The C++ side could not be built or run; the message says what to do."""


def check_toolchain() -> None:
    if shutil.which("cmake") is None:
        raise CppError(
            "cmake not found in PATH. Install it (macOS: `brew install cmake`) and try again."
        )
    compiler = os.environ.get("CXX") or next(
        (c for c in ("c++", "clang++", "g++") if shutil.which(c)), None
    )
    if compiler is None or shutil.which(compiler) is None:
        raise CppError(
            "no C++ compiler found (looked for $CXX, c++, clang++, g++). "
            "On macOS run `xcode-select --install`."
        )


def _run(cmd: list[str], what: str) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise CppError(f"{what}: cannot execute {cmd[0]}: {exc}") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-25:])
        raise CppError(f"{what} failed (exit {proc.returncode}). Last output:\n{tail}")
    return proc.stdout


def build() -> None:
    """Configure and compile in Release. Eigen is fetched on the first configure (needs network)."""
    check_toolchain()
    _run(
        ["cmake", "-S", str(CPP_DIR), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"],
        "cmake configure (the first run downloads Eigen, check your connection)",
    )
    _run(["cmake", "--build", str(BUILD_DIR), "--config", "Release"], "cmake build")


def binary_path():
    for candidate in (BUILD_DIR / "nn_eigen", BUILD_DIR / "Release" / "nn_eigen.exe"):
        if candidate.exists():
            return candidate
    raise CppError(f"binary not found in {BUILD_DIR}; run build() first")


def run() -> dict:
    """Run the full training in C++ and return its results.

    The old output files are deleted first, so stale results can never be loaded.
    """
    for stale in (CPP_RESULTS_JSON, CPP_LOSS_CURVE_CSV):
        stale.unlink(missing_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    _run(
        [str(binary_path()), "--train", str(config.DATA_DIR), "--out", str(config.RESULTS_DIR)],
        "C++ training run",
    )
    return load_results()


def load_results() -> dict:
    if not CPP_RESULTS_JSON.exists():
        raise CppError(f"{CPP_RESULTS_JSON} not found; the C++ binary did not write results")
    return json.loads(CPP_RESULTS_JSON.read_text())
