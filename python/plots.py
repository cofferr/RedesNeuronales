"""Generate comparison plots and tables from the files in results/.

Figures go to results/figures/ as PNG (200 dpi) and PDF; the summary table goes to
results/summary_table.csv and results/summary_table.md. Everything is built from the
result files, nothing is hard-coded. Styling: two categorical hues (validated palette
slots 1-2) that also differ in line style / hatch, so they stay distinguishable in grayscale.
"""

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.datasets import load_iris  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402

from . import config  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
C_CPP, C_SK = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated)
CLASS_NAMES = list(load_iris().target_names)
COLUMN = 3.3  # inches: one column
PAGE = 6.5  # inches: full text width

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


# ---------------------------------------------------------------- inputs

def load_inputs(results_dir: Path) -> dict:
    def read_json(name):
        path = results_dir / name
        return json.loads(path.read_text()) if path.exists() else None

    return {
        "sk": read_json("sklearn_results.json"),
        "cpp": read_json("cpp_results.json"),
        "bench": read_json("benchmark.json"),
        "sk_curve": pd.read_csv(results_dir / "sklearn_loss_curve.csv"),
        "cpp_curve": pd.read_csv(results_dir / "cpp_loss_curve.csv"),
        "sk_pred": pd.read_csv(results_dir / "sklearn_predictions.csv"),
        "cpp_pred": pd.read_csv(results_dir / "cpp_predictions.csv"),
    }


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _note(fig, text: str, width: int = 95, y: float = -0.02) -> None:
    fig.text(0.0, y, "\n".join(textwrap.wrap(text, width)), ha="left", va="top",
             fontsize=6.5, color=INK2)


# ---------------------------------------------------------------- figures

def fig_loss_curves(d: dict, out_dir: Path) -> list[Path]:
    sk, cpp = d["sk_curve"], d["cpp_curve"]
    diff = (sk["loss"] - cpp["loss"]).abs().to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(PAGE, 2.2), constrained_layout=True)

    for ax, log in zip(axes[:2], (False, True)):
        ax.plot(cpp["epoch"], cpp["loss"], color=C_CPP, lw=1.8, ls="-", label="C++ (Eigen)")
        ax.plot(sk["epoch"], sk["loss"], color=C_SK, lw=1.1, ls="--", marker="o", ms=3,
                markevery=(50, 100), label="scikit-learn")
        ax.set_xlabel("Epoch")
        if log:
            ax.set_yscale("log")
    axes[0].set_ylabel("Training loss (cross-entropy)")
    axes[0].set_title("(a) Linear scale")
    axes[1].set_title("(b) Log scale")
    axes[0].legend(loc="upper right")

    # Exact zeros cannot be drawn on a log axis: plot only the nonzero differences as dots.
    nonzero = diff > 0
    axes[2].plot(cpp["epoch"][nonzero], diff[nonzero], ls="none", marker=".", ms=1.6, color=INK)
    axes[2].axhline(np.finfo(float).eps, color=INK2, lw=0.8, ls=":")
    axes[2].text(config.EPOCHS, np.finfo(float).eps * 1.6, "float64 ε = 2.2e-16",
                 ha="right", va="bottom", fontsize=6.5, color=INK2)
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-18, 1e-12)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("|sklearn − C++|")
    axes[2].set_title("(c) Difference, log scale")
    axes[2].text(0.97, 0.95, f"max = {diff.max():.1e}\n{int((~nonzero).sum())} epochs exactly 0",
                 transform=axes[2].transAxes, ha="right", va="top", fontsize=6.5, color=INK)
    return _save(fig, out_dir, "fig_loss_curves")


def fig_confusion(d: dict, out_dir: Path) -> list[Path]:
    pred = d["sk_pred"]
    cm = confusion_matrix(pred["true"], pred["pred"], labels=range(len(CLASS_NAMES)))
    fig, ax = plt.subplots(figsize=(COLUMN, 3.0))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix, test set (n={cm.sum()})")
    for spine in ax.spines.values():
        spine.set_visible(False)
    _note(fig, "C++ and scikit-learn give identical test predictions "
               "(checked in every run and by tests/test_equivalence.py).", width=62, y=-0.02)
    return _save(fig, out_dir, "fig_confusion_matrix")


def _bar_labels(ax, xs, tops, texts, log=False):
    for x, top, text in zip(xs, tops, texts):
        ax.text(x, top * (1.12 if log else 1.0) + (0 if log else 0.0), text,
                ha="center", va="bottom", fontsize=7, color=INK)


def fig_time(d: dict, out_dir: Path) -> list[Path] | None:
    if d["bench"] is None:
        return None
    t = d["bench"]["time_s"]
    ov = d["bench"]["python_overhead_partial_fit"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(COLUMN + 1.0, 2.9), sharey=True, constrained_layout=True,
        gridspec_kw={"width_ratios": [2, 1]})

    def bars(ax, xs, stats_list, colors, hatches):
        for x, s, color, hatch in zip(xs, stats_list, colors, hatches):
            ms = {k: s[k] * 1e3 for k in ("median", "min", "max")}
            ax.bar(x, ms["median"], width=0.62, color=color, edgecolor=INK, linewidth=0.7,
                   hatch=hatch)
            ax.errorbar(x, ms["median"], yerr=[[ms["median"] - ms["min"]],
                                                [ms["max"] - ms["median"]]],
                        color=INK, capsize=3, lw=0.9)
            ax.text(x, ms["max"] * 1.12, f"{ms['median']:.1f} ms", ha="center", va="bottom",
                    fontsize=7)

    bars(ax1, [0, 1], [t["cpp"], t["sklearn_fit"]], [C_CPP, C_SK], ["", "///"])
    ax1.set_xticks([0, 1], ["C++\n(Eigen)", "scikit-learn\nfit()"])
    ax1.set_ylabel("Training time (ms, log scale)")
    ax1.set_title("Training, 1000 epochs")
    bars(ax2, [0], [ov], [C_SK], ["xxx"])
    ax2.set_xticks([0], ["scikit-learn\npartial_fit loop"])
    ax2.set_title("Python overhead")
    for ax in (ax1, ax2):
        ax.set_yscale("log")
        ax.set_ylim(1, 3000)
        ax.set_xlim(-0.6, 1.6 if ax is ax1 else 0.6)
    ax1.text(0.5, 0.97, f"median speedup {t['speedup_median']:.1f}×", transform=ax1.transAxes,
             ha="center", va="top", fontsize=7.5, color=INK)
    _note(fig, f"Bars: median; whiskers: min–max over {t['cpp']['n']} runs per side "
               "(interleaved CSSC CSSC blocks, warm-up discarded). The partial_fit loop calls "
               "Python once per epoch; it measures call overhead, not the algorithm, and is "
               "not part of the speedup.", width=62, y=-0.02)
    return _save(fig, out_dir, "fig_time")


def fig_memory(d: dict, out_dir: Path) -> list[Path] | None:
    if d["bench"] is None:
        return None
    m = d["bench"]["memory"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLUMN + 1.0, 2.7), constrained_layout=True)
    names = ["C++\n(Eigen)", "scikit-\nlearn"]
    for ax, key, scale, unit, title, log in (
        (ax1, "peak_rss_total_bytes", 1e6, "MB", "Peak memory (whole process)", True),
        (ax2, "train_increment_bytes", 1e3, "KB", "Growth during training", False),
    ):
        for x, side, color, hatch in ((0, "cpp", C_CPP, ""), (1, "sklearn", C_SK, "///")):
            s = m[side][key]
            med, lo, hi = (s[k] / scale for k in ("median", "min", "max"))
            ax.bar(x, med, width=0.62, color=color, edgecolor=INK, linewidth=0.7, hatch=hatch)
            ax.errorbar(x, med, yerr=[[med - lo], [hi - med]], color=INK, capsize=3, lw=0.9)
            ax.text(x, hi * (1.12 if log else 1.0) + (0 if log else 3), f"{med:.3g} {unit}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks([0, 1], names)
        ax.set_title(title)
        ax.set_ylabel(f"{unit}" + (" (log scale)" if log else ""))
        ax.set_xlim(-0.6, 1.6)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(0.5, 1000)
        else:
            ax.set_ylim(0, 80)
    runs = d["bench"]["method"]["memory_runs_per_side"]
    _note(fig, "Peak = maximum resident set size of the whole process: for C++ the binary, "
               "Eigen and the data; for scikit-learn the Python interpreter, numpy, scipy, "
               "scikit-learn and pandas (scikit-learn 1.9.1 imports pandas itself, so a "
               "pandas-free peak could not be measured). Growth = increase of that peak while "
               "training (RSS is counted in 16 KB pages, so 0 means no new page). "
               f"Median and min–max of {runs} clean processes per side.",
          width=62, y=-0.02)
    return _save(fig, out_dir, "fig_memory")


# ---------------------------------------------------------------- tables

def _class_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    p, r, f, _ = precision_recall_fscore_support(
        pred["true"], pred["pred"], labels=range(len(CLASS_NAMES)), zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f}, index=CLASS_NAMES)


def build_table(d: dict) -> tuple[list[tuple], list[str]]:
    """Rows (section, metric, C++, scikit-learn) as formatted strings, plus notes."""
    sk, cpp = d["sk"], d["cpp"]
    rows = [
        ("Accuracy and loss", "Train accuracy", f"{cpp['train_accuracy']:.4f}", f"{sk['train_accuracy']:.4f}"),
        ("Accuracy and loss", "Test accuracy", f"{cpp['test_accuracy']:.4f}", f"{sk['test_accuracy']:.4f}"),
        ("Accuracy and loss", "Train loss", f"{cpp['train_loss']:.6f}", f"{sk['train_loss']:.6f}"),
        ("Accuracy and loss", "Test loss", f"{cpp['test_loss']:.6f}", f"{sk['test_loss']:.6f}"),
        ("Accuracy and loss", "Max absolute loss-curve difference", "—",
         f"{(d['sk_curve']['loss'] - d['cpp_curve']['loss']).abs().max():.2e}"),
    ]
    cm_cpp, cm_sk = _class_metrics(d["cpp_pred"]), _class_metrics(d["sk_pred"])
    for cls in CLASS_NAMES:
        for metric in ("precision", "recall", "f1"):
            rows.append(("Test metrics per class", f"{cls} {metric}",
                         f"{cm_cpp.loc[cls, metric]:.4f}", f"{cm_sk.loc[cls, metric]:.4f}"))
    notes = [
        "C++ and scikit-learn follow the same trajectory: same data, same initial weights, plain "
        "full-batch gradient descent, float64.",
    ]
    b = d["bench"]
    if b is not None:
        t, mem = b["time_s"], b["memory"]
        n = t["cpp"]["n"]
        for stat in ("median", "mean", "std", "min", "max"):
            rows.append((f"Training time, ms (n={n} per side)", stat,
                         f"{t['cpp'][stat] * 1e3:.2f}", f"{t['sklearn_fit'][stat] * 1e3:.2f}"))
        rows.append((f"Training time, ms (n={n} per side)", "median speedup over scikit-learn",
                     f"{t['speedup_median']:.1f}×", "1× (baseline)"))
        ov = b["python_overhead_partial_fit"]
        rows.append(("Python overhead, ms", "partial_fit loop, median (min–max)", "—",
                     f"{ov['median'] * 1e3:.1f} ({ov['min'] * 1e3:.1f}–{ov['max'] * 1e3:.1f})"))
        runs = b["method"]["memory_runs_per_side"]
        for key, label, scale, unit in (("peak_rss_total_bytes", "Peak RSS", 1e6, "MB"),
                                        ("train_increment_bytes", "Peak growth during training", 1e3, "KB")):
            fmt = lambda s: f"{s['median'] / scale:.3g} {unit} ({s['min'] / scale:.3g}–{s['max'] / scale:.3g})"
            rows.append((f"Memory (median and min–max, {runs} processes)", label,
                         fmt(mem["cpp"][key]), fmt(mem["sklearn"][key])))
        env = b["environment"]
        notes += [
            "scikit-learn time = one fit() call with warm_start=True starting from the shared initial "
            "weights (no re-initialization), so it runs the same 1000 epochs as C++. Model creation "
            "and weight loading are outside the timed region; C++ times only the training loop.",
            "The partial_fit loop (one call per epoch) is Python call overhead and is reported "
            "separately; it is not part of the speedup.",
            "scikit-learn's peak RSS includes pandas: scikit-learn 1.9.1 imports it itself when it is "
            "installed (sklearn/utils/fixes.py), and blocking it breaks scikit-learn, so a "
            "pandas-free peak could not be measured. It also includes the interpreter, numpy and scipy.",
            "Loss curves are recorded at the same point (forward pass before each update). "
            f"Test predictions are identical, so per-class precision/recall/F1 coincide.",
            f"Environment: {env['cpu']} ({env['logical_cpus']} logical CPUs, "
            f"{env['ram_bytes'] / 2**30:.0f} GiB), {env['platform']}; {env['cpp_compiler']} "
            f"flags `{env['cpp_flags']}` (CMake {env['cmake_build_type']}); Eigen {env['eigen']}; "
            f"Python {env['python']}, numpy {env['numpy']}, scikit-learn {env['scikit_learn']}; "
            f"{env['power_source']}.",
        ]
    return rows, notes


def write_tables(d: dict, results_dir: Path) -> list[Path]:
    rows, notes = build_table(d)
    csv_path, md_path = results_dir / "summary_table.csv", results_dir / "summary_table.md"
    pd.DataFrame(rows, columns=["section", "metric", "cpp_eigen", "scikit_learn"]).to_csv(
        csv_path, index=False)
    lines = ["# Summary: C++ (Eigen) vs scikit-learn", ""]
    section = None
    for sec, metric, a, b in rows:
        if sec != section:
            if section is not None:
                lines.append("")
            lines += [f"**{sec}**", "", "| Metric | C++ (Eigen) | scikit-learn |", "|---|---:|---:|"]
            section = sec
        lines.append(f"| {metric} | {a} | {b} |")
    lines += ["", "**Notes**", ""] + [f"{i}. {n}" for i, n in enumerate(notes, 1)]
    md_path.write_text("\n".join(lines) + "\n")
    return [csv_path, md_path]


# ---------------------------------------------------------------- entry point

def generate_all(results_dir: Path = config.RESULTS_DIR, out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or results_dir / "figures"
    d = load_inputs(results_dir)
    written: list[Path] = []
    for fn in (fig_loss_curves, fig_confusion, fig_time, fig_memory):
        written += fn(d, out_dir) or []
    written += write_tables(d, results_dir)
    return written
