"""Benchmarks: C++ vs scikit-learn training time and memory.

Time: 8 blocks of 5 measured repetitions per side (20 in total per side), ordered
C S S C C S S C (ABBA) so slow thermal/frequency drift hits both sides equally. Each block
starts with 2 discarded warm-up runs. C++ blocks are one ``nn_eigen --bench`` process each
(repetitions inside the process, only the training loop timed); sklearn blocks run in this
process (a new MLPClassifier per repetition, only the fit() call timed). Every repetition
checks that the final train loss matches the reference.

Memory: separate clean processes (one training each), several runs per side, peak RSS
measured externally with wait4 plus the training increment reported by the process itself.
"""

import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import sklearn

from . import config, cpp_runner, data, sklearn_model

BLOCK_ORDER = "CSSCCSSC"
BLOCK_REPS = 5
WARMUP = 2
MEM_RUNS = 5
LOSS_TOL = 1e-9
BENCHMARK_JSON = config.RESULTS_DIR / "benchmark.json"
BLOCK_JSON = config.RESULTS_DIR / "cpp_bench_block.json"


def stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def rss_unit() -> int:
    """Bytes per ru_maxrss unit: bytes on macOS, kilobytes on Linux."""
    return 1 if sys.platform == "darwin" else 1024


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cmake_cache() -> dict[str, str]:
    cache = {}
    path = cpp_runner.BUILD_DIR / "CMakeCache.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.startswith(("//", "#")):
                key, value = line.split("=", 1)
                cache[key.split(":")[0]] = value
    return cache


def _compile_flags() -> str:
    """Flags CMake really passes to the compiler (from flags.make), not the cache defaults."""
    path = cpp_runner.BUILD_DIR / "CMakeFiles" / "nn_eigen.dir" / "flags.make"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("CXX_FLAGS ="):
                return line.split("=", 1)[1].strip()
    return ""


def environment(cpp_block: dict) -> dict:
    cache = _cmake_cache()
    compiler_path = cache.get("CMAKE_CXX_COMPILER", "")
    try:
        compiler_version = subprocess.run(
            [compiler_path, "--version"], capture_output=True, text=True
        ).stdout.splitlines()[0]
    except (OSError, IndexError):
        compiler_version = cpp_block.get("compiler")
    try:
        power = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True
        ).stdout.splitlines()[0]
    except (OSError, IndexError):
        power = None
    mem = _sysctl("hw.memsize")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "model": _sysctl("hw.model"),
        "logical_cpus": os.cpu_count(),
        "ram_bytes": int(mem) if mem else None,
        "power_source": power,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "eigen": cpp_block.get("eigen_version"),
        "cpp_compiler": compiler_version,
        "cpp_compiler_path": compiler_path,
        "cpp_standard": 17,
        "cmake_build_type": cache.get("CMAKE_BUILD_TYPE"),
        "cpp_flags": _compile_flags(),
        "epochs": config.EPOCHS,
        "learning_rate": config.LEARNING_RATE,
    }


def _check_loss(loss: float, reference: float, who: str) -> None:
    if abs(loss - reference) > LOSS_TOL:
        raise RuntimeError(f"{who}: final loss {loss} differs from the reference {reference}")


def cpp_block(reference: float) -> dict:
    cpp_runner._run(
        [str(cpp_runner.binary_path()), "--bench", str(config.DATA_DIR),
         "--out", str(config.RESULTS_DIR), "--warmup", str(WARMUP),
         "--repeat", str(BLOCK_REPS), "--ref-loss", repr(reference)],
        "C++ benchmark block",
    )
    block = json.loads(BLOCK_JSON.read_text())
    for loss in block["final_train_losses"]:
        _check_loss(loss, reference, "C++")
    return block


def sklearn_block(X, y, reference: float, runner=sklearn_model.timed_fit) -> list[float]:
    """WARMUP discarded + BLOCK_REPS measured; a NEW model per repetition, only fit() timed."""
    times = []
    for r in range(WARMUP + BLOCK_REPS):
        clf = sklearn_model.prepare_single_fit(X, y)  # setup, not timed
        elapsed = runner(clf, X, y)
        _check_loss(sklearn_model.cross_entropy(clf, X, y), reference, "sklearn")
        if r >= WARMUP:
            times.append(elapsed)
    return times


def partial_fit_overhead(X, y, reference: float) -> dict:
    """Separate section: the per-epoch partial_fit loop, i.e. Python call overhead."""
    times = []
    for r in range(WARMUP + 20):
        clf, elapsed = sklearn_model.train_partial_fit(X, y)
        _check_loss(sklearn_model.cross_entropy(clf, X, y), reference, "sklearn partial_fit")
        if r >= WARMUP:
            times.append(elapsed)
    return {"times_s": times, **stats(times)}


def _measure_process(cmd: list[str]) -> tuple[dict, int]:
    """Run cmd in a clean process; return its JSON stdout line and its external peak RSS."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, cwd=config.ROOT)
    out = proc.stdout.read()
    _, status, usage = os.wait4(proc.pid, 0)  # per-process rusage, unlike RUSAGE_CHILDREN
    proc.stdout.close()
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd} failed with exit code {proc.returncode}")
    return json.loads(out.strip().splitlines()[-1]), usage.ru_maxrss * rss_unit()


def memory(reference: float) -> dict:
    """Clean processes alternated C++/Python so both see the same machine state."""
    runs = {"cpp": [], "sklearn": []}
    commands = {
        "cpp": [str(cpp_runner.binary_path()), "--mem", str(config.DATA_DIR)],
        "sklearn": [sys.executable, "-m", "python.bench_worker"],
    }
    for _ in range(MEM_RUNS):
        for side in ("cpp", "sklearn"):
            own, external_peak = _measure_process(commands[side])
            runs[side].append({**own, "peak_rss_total_bytes": external_peak})
    return {
        side: {
            "runs": r,
            "peak_rss_total_bytes": stats([x["peak_rss_total_bytes"] for x in r]),
            "train_increment_bytes": stats([x["train_increment_bytes"] for x in r]),
            "rss_before_train_bytes": stats([x["rss_before_train_bytes"] for x in r]),
        }
        for side, r in runs.items()
    }


def run() -> dict:
    reference = json.loads((config.RESULTS_DIR / "sklearn_results.json").read_text())["train_loss"]
    X, _, y, _ = data.load_csv()
    times = {"C": [], "S": []}
    blocks = []
    first_cpp_block = None
    for index, side in enumerate(BLOCK_ORDER):
        if side == "C":
            block = cpp_block(reference)
            first_cpp_block = first_cpp_block or block
            block_times = block["times_s"]
        else:
            block_times = sklearn_block(X, y, reference)
        times[side] += block_times
        blocks.append({"index": index, "side": "cpp" if side == "C" else "sklearn",
                       "median_s": statistics.median(block_times)})

    cpp_stats, sk_stats = stats(times["C"]), stats(times["S"])
    result = {
        "environment": environment(first_cpp_block),
        "method": {
            "block_order": BLOCK_ORDER, "reps_per_block": BLOCK_REPS,
            "warmup_per_block": WARMUP, "measured_reps_per_side": len(times["C"]),
            "reference_train_loss": reference, "loss_tolerance": LOSS_TOL,
            "memory_runs_per_side": MEM_RUNS,
        },
        "time_s": {
            "cpp": {"times_s": times["C"], **cpp_stats},
            "sklearn_fit": {"times_s": times["S"], **sk_stats},
            "speedup_median": sk_stats["median"] / cpp_stats["median"],
            "blocks": blocks,
        },
        "python_overhead_partial_fit": partial_fit_overhead(X, y, reference),
        "memory": memory(reference),
    }
    BENCHMARK_JSON.write_text(json.dumps(result, indent=2))
    return result


def summary(result: dict) -> str:
    t, m = result["time_s"], result["memory"]
    mb = lambda b: b / 1e6  # noqa: E731
    lines = [f"{'time (ms)':<22}{'median':>9}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}"]
    for name, key in (("C++ (Eigen)", "cpp"), ("sklearn fit()", "sklearn_fit")):
        s = t[key]
        lines.append(f"{name:<22}" + "".join(f"{s[k] * 1e3:>9.2f}" for k in
                                            ("median", "mean", "std", "min", "max")))
    o = result["python_overhead_partial_fit"]
    lines.append(f"{'sklearn partial_fit*':<22}" + "".join(f"{o[k] * 1e3:>9.2f}" for k in
                                                            ("median", "mean", "std", "min", "max")))
    lines.append(f"speedup (median sklearn fit() / C++): {t['speedup_median']:.1f}x   "
                 f"(*partial_fit loop = Python call overhead, not part of the speedup)")
    lines.append(f"{'memory (MB)':<22}{'peak total':>12}{'train incr.':>13}")
    for name, key in (("C++ (Eigen)", "cpp"), ("sklearn fit()", "sklearn")):
        lines.append(f"{name:<22}{mb(m[key]['peak_rss_total_bytes']['median']):>12.1f}"
                     f"{mb(m[key]['train_increment_bytes']['median']):>13.3f}")
    return "\n".join(lines)
