"""Generate everything the LaTeX report needs from the pipeline results.

Writes report/generated/numbers.tex (one macro per figure quoted in the text) and the
booktabs tables, copies the PDF figures to report/figures/, and can pack the report folder
into a zip for Overleaf. main.tex contains no hand-written numbers: every figure comes from
here, which in turn reads benchmark.json and the other result files.

Usage: ``uv run python -m python.report_data [--zip]`` (run_all.py calls generate()).
"""

import ast
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from . import config, cpp_runner, plots, sklearn_model

REPORT_DIR = config.ROOT / "report"
GENERATED_DIR = REPORT_DIR / "generated"
FIGURES_DIR = REPORT_DIR / "figures"
ZIP_PATH = config.ROOT / "report_overleaf.zip"
PY_DIR = config.ROOT / "python"
CPP_DIR = config.ROOT / "cpp" / "src"

_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}"}


def tex(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    return "".join(_TEX_SPECIAL.get(c, c) for c in str(text))


def sci(value: float, digits: int = 1) -> str:
    """Scientific notation as a LaTeX math expression, e.g. 6.7e-16 -> 6.7\\times10^{-16}."""
    if value == 0:
        return r"\ensuremath{0}"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"\ensuremath{{{mantissa}\times10^{{{int(exponent)}}}}}"


def fixed(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


# ------------------------------------------------------------------ checks from the binary

def run_binary_checks() -> dict:
    """Run nn_eigen's own verifications and parse their numbers (nothing is typed by hand)."""
    binary = str(cpp_runner.binary_path())
    out = subprocess.run([binary, str(config.DATA_DIR)], capture_output=True, text=True,
                         check=True).stdout
    grad = subprocess.run([binary, "--gradcheck", str(config.DATA_DIR)], capture_output=True,
                          text=True, check=True).stdout

    def diff_of(label: str) -> float:
        match = re.search(rf"{re.escape(label)}.*?\|diff\| = ([0-9.e+-]+)", out)
        if not match:
            raise RuntimeError(f"could not find '{label}' in the nn_eigen check output")
        return float(match.group(1))

    rel = [float(x) for x in re.findall(r"relative error ([0-9.e+-]+)", grad)]
    if len(rel) != 8 or "gradient check: PASS" not in grad:
        raise RuntimeError("unexpected nn_eigen --gradcheck output")
    return {
        "init_train_loss_diff": diff_of("initial train loss"),
        "init_test_loss_diff": diff_of("initial test loss"),
        "proba_diff": max(diff_of("max |proba - sklearn| (train)"),
                          diff_of("max |proba - sklearn| (test)")),
        "weights_diff": diff_of("weights after"),
        "curve20_diff": diff_of("loss curve ("),
        "gradcheck_max_rel": max(rel),
        "gradcheck_points": len(rel) // 4,
    }


# ------------------------------------------------------------------ lines of code

def _code_lines(source: str, drop: set[int] = frozenset()) -> int:
    """Non-blank, non-comment lines (1-based line numbers in `drop` are ignored)."""
    count, in_block = 0, False
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if number in drop or not stripped:
            continue
        if in_block:
            in_block = "*/" not in stripped
            continue
        if stripped.startswith("/*"):
            in_block = "*/" not in stripped
            continue
        if stripped.startswith(("//", "#")):
            continue
        count += 1
    return count


def sloc_file(path: Path) -> int:
    return _code_lines(path.read_text())


def sloc_function(path: Path, name: str) -> int:
    """Code lines of one Python function, docstring excluded."""
    source = path.read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            lines = source.splitlines()[node.lineno - 1: node.end_lineno]
            doc = node.body[0]
            drop = set()
            if isinstance(doc, ast.Expr) and isinstance(getattr(doc, "value", None), ast.Constant) \
                    and isinstance(doc.value.value, str):
                drop = set(range(doc.lineno - node.lineno + 1, doc.end_lineno - node.lineno + 2))
            return _code_lines("\n".join(lines), drop)
    raise KeyError(f"{name} not found in {path}")


def loc_rows() -> list[tuple[str, int | None, int | None]]:
    """(component, C++ lines, Python lines); None means 'not needed on that side'."""
    sk = PY_DIR / "sklearn_model.py"
    rows = [
        ("Definición del modelo y entrenamiento",
         sloc_file(CPP_DIR / "network.hpp"), sloc_function(sk, "build_model")),
        ("Lectura de datos y pesos desde CSV",
         sloc_file(CPP_DIR / "csv.hpp"),
         sloc_function(PY_DIR / "data.py", "load_csv")
         + sloc_function(PY_DIR / "weights.py", "load_initial_weights")),
        ("Verificación numérica de gradientes", sloc_file(CPP_DIR / "gradcheck.hpp"), None),
        ("Equivalencia con C++ (pesos iniciales, warm\\_start)", None,
         sloc_function(sk, "load_initial_into") + sloc_function(sk, "prepare_single_fit")),
    ]
    return rows


# ------------------------------------------------------------------ LaTeX helpers

def latex_table(header: list[str], sections: list[tuple[str | None, list[list[str]]]],
                colspec: str) -> str:
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\"]
    for title, rows in sections:
        lines.append(r"\midrule")
        if title:
            lines.append(rf"\multicolumn{{{len(header)}}}{{l}}{{\textit{{{title}}}}} \\")
        lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _write(name: str, text: str) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    (GENERATED_DIR / name).write_text(text)


# ------------------------------------------------------------------ macros

def build_macros(d: dict, checks: dict, history: list[dict]) -> dict[str, str]:
    sk, cpp, b = d["sk"], d["cpp"], d["bench"]
    t, mem, env = b["time_s"], b["memory"], b["environment"]
    method = b["method"]
    true = d["sk_pred"]["true"].to_numpy()
    pred = d["sk_pred"]["pred"].to_numpy()
    blocks = t["blocks"]
    c_blocks = [x["median_s"] for x in blocks if x["side"] == "cpp"]
    s_blocks = [x["median_s"] for x in blocks if x["side"] == "sklearn"]
    speeds = [h["speedup_median"] for h in history]
    ov = b["python_overhead_partial_fit"]
    diff = (d["sk_curve"]["loss"] - d["cpp_curve"]["loss"]).abs()
    ms = lambda s: fixed(s * 1e3, 2)  # noqa: E731
    m = {
        # data and configuration
        "nTotal": str(config.N_TRAIN + len(true)), "nTrain": str(config.N_TRAIN),
        "nTest": str(len(true)), "nFeatures": str(config.N_FEATURES),
        "nClasses": str(config.N_CLASSES), "nHidden": str(config.N_HIDDEN),
        "learningRate": str(config.LEARNING_RATE), "nEpochs": str(config.EPOCHS),
        "checkEpochs": str(config.CHECK_EPOCHS), "seedValue": str(config.SEED),
        "testFraction": f"{int(config.TEST_SIZE * 100)}\\%",
        "nParams": str(config.N_FEATURES * config.N_HIDDEN + config.N_HIDDEN
                       + config.N_HIDDEN * config.N_CLASSES + config.N_CLASSES),
        # quality
        "cppTrainAcc": fixed(cpp["train_accuracy"] * 100, 2) + "\\%",
        "cppTestAcc": fixed(cpp["test_accuracy"] * 100, 2) + "\\%",
        "skTrainAcc": fixed(sk["train_accuracy"] * 100, 2) + "\\%",
        "skTestAcc": fixed(sk["test_accuracy"] * 100, 2) + "\\%",
        "cppTrainLoss": fixed(cpp["train_loss"], 4), "skTrainLoss": fixed(sk["train_loss"], 4),
        "cppTestLoss": fixed(cpp["test_loss"], 4), "skTestLoss": fixed(sk["test_loss"], 4),
        "initTrainLoss": fixed(d["sk_curve"]["loss"].iloc[0], 4),
        "testErrors": str(int((true != pred).sum())),
        "testCorrect": str(int((true == pred).sum())),
        "maxLossDiff": sci(float(diff.max())),
        "zeroDiffEpochs": str(int((diff == 0).sum())),
        # checks performed by the binary
        "initLossDiff": sci(max(checks["init_train_loss_diff"], checks["init_test_loss_diff"])),
        "probaDiff": sci(checks["proba_diff"]),
        "weightsDiff": sci(checks["weights_diff"]),
        "curveTwentyDiff": sci(checks["curve20_diff"]),
        "gradcheckMaxRel": sci(checks["gradcheck_max_rel"]),
        "gradcheckPoints": str(checks["gradcheck_points"]),
        # timing
        "nReps": str(t["cpp"]["n"]), "nWarmup": str(method["warmup_per_block"]),
        "nBlocks": str(len(blocks)), "repsPerBlock": str(method["reps_per_block"]),
        "blockOrder": r"\texttt{" + " ".join(method["block_order"]) + "}",
        "cppMedianMs": ms(t["cpp"]["median"]), "skMedianMs": ms(t["sklearn_fit"]["median"]),
        "cppMeanMs": ms(t["cpp"]["mean"]), "skMeanMs": ms(t["sklearn_fit"]["mean"]),
        "cppStdMs": ms(t["cpp"]["std"]), "skStdMs": ms(t["sklearn_fit"]["std"]),
        "cppMinMs": ms(t["cpp"]["min"]), "skMinMs": ms(t["sklearn_fit"]["min"]),
        "cppMaxMs": ms(t["cpp"]["max"]), "skMaxMs": ms(t["sklearn_fit"]["max"]),
        "speedupMedian": fixed(t["speedup_median"], 1),
        "speedupBlockMin": fixed(min(s_blocks) / max(c_blocks), 1),
        "speedupBlockMax": fixed(max(s_blocks) / min(c_blocks), 1),
        "speedupHistMin": fixed(min(speeds), 1), "speedupHistMax": fixed(max(speeds), 1),
        "nHistoryRuns": str(len(history)),
        "partialMedianMs": fixed(ov["median"] * 1e3, 1),
        "partialOverFit": fixed(ov["median"] / t["sklearn_fit"]["median"], 1),
        "cppPerEpochUs": fixed(t["cpp"]["median"] / config.EPOCHS * 1e6, 1),
        "skPerEpochUs": fixed(t["sklearn_fit"]["median"] / config.EPOCHS * 1e6, 1),
        # memory
        "nMemRuns": str(method["memory_runs_per_side"]),
        "cppPeakMB": fixed(mem["cpp"]["peak_rss_total_bytes"]["median"] / 1e6, 2),
        "skPeakMB": fixed(mem["sklearn"]["peak_rss_total_bytes"]["median"] / 1e6, 0),
        "memRatio": fixed(mem["sklearn"]["peak_rss_total_bytes"]["median"]
                          / mem["cpp"]["peak_rss_total_bytes"]["median"], 0),
        "cppGrowthKB": fixed(mem["cpp"]["train_increment_bytes"]["median"] / 1e3, 1),
        "skGrowthKB": fixed(mem["sklearn"]["train_increment_bytes"]["median"] / 1e3, 0),
        "workspaceKB": fixed(workspace_bytes() / 1e3, 0),
        # environment
        "envCpu": tex(env["cpu"]), "envCpus": str(env["logical_cpus"]),
        "envRamGiB": fixed(env["ram_bytes"] / 2**30, 0), "envOs": tex(env["platform"]),
        "envCompiler": tex(env["cpp_compiler"]), "envFlags": r"\texttt{" + tex(env["cpp_flags"]) + "}",
        "envEigen": tex(env["eigen"]), "envPython": tex(env["python"]),
        "envNumpy": tex(env["numpy"]), "envSklearn": tex(env["scikit_learn"]),
        "envPower": tex(env["power_source"] or "n/d"),
    }
    return m


def workspace_bytes() -> int:
    """Size in bytes of the buffers preallocated per epoch (Workspace + Grads), float64."""
    n, nin, nh, no = config.N_TRAIN, config.N_FEATURES, config.N_HIDDEN, config.N_CLASSES
    # Z1, H, dH, dZ1 (n x hidden); logits, log_probs, probs, dZ2 (n x classes); row_max, log_sum
    workspace = 4 * n * nh + 4 * n * no + 2 * n
    grads = nin * nh + nh + nh * no + no
    return 8 * (workspace + grads)


def write_numbers(macros: dict[str, str]) -> None:
    lines = ["% Generated by python/report_data.py -- do not edit by hand."]
    lines += [rf"\newcommand{{\{k}}}{{{v}\xspace}}" for k, v in macros.items()]
    _write("numbers.tex", "\n".join(lines) + "\n")


# ------------------------------------------------------------------ tables

def write_tables(d: dict, macros: dict[str, str]) -> None:
    sk, cpp, b = d["sk"], d["cpp"], d["bench"]
    t, mem = b["time_s"], b["memory"]
    ms = lambda s: fixed(s * 1e3, 2)  # noqa: E731

    quality = [
        ["Exactitud (entrenamiento)", macros["cppTrainAcc"], macros["skTrainAcc"]],
        ["Exactitud (prueba)", macros["cppTestAcc"], macros["skTestAcc"]],
        ["Pérdida (entrenamiento)", fixed(cpp["train_loss"], 6), fixed(sk["train_loss"], 6)],
        ["Pérdida (prueba)", fixed(cpp["test_loss"], 6), fixed(sk["test_loss"], 6)],
    ]
    time_rows = [[label, ms(t["cpp"][k]), ms(t["sklearn_fit"][k])]
                 for label, k in (("Mediana", "median"), ("Media", "mean"),
                                  ("Desv. estándar", "std"), ("Mínimo", "min"), ("Máximo", "max"))]
    time_rows.append(["Aceleración (razón de medianas)", macros["speedupMedian"] + r"$\times$",
                      "1$\\times$"])
    ov = b["python_overhead_partial_fit"]
    time_rows.append([r"Bucle \texttt{partial\_fit}, mediana (mín--máx)", "---",
                      f"{ov['median'] * 1e3:.1f} ({ov['min'] * 1e3:.1f}--{ov['max'] * 1e3:.1f})"])

    def mem_cell(side: str, key: str, scale: float, digits: int) -> str:
        s = mem[side][key]
        return (f"{s['median'] / scale:.{digits}f} ({s['min'] / scale:.{digits}f}"
                f"--{s['max'] / scale:.{digits}f})")

    mem_rows = [
        ["Pico de RSS (MB)", mem_cell("cpp", "peak_rss_total_bytes", 1e6, 2),
         mem_cell("sklearn", "peak_rss_total_bytes", 1e6, 0)],
        ["Crecimiento del pico en el entrenamiento (KB)",
         mem_cell("cpp", "train_increment_bytes", 1e3, 1),
         mem_cell("sklearn", "train_increment_bytes", 1e3, 0)],
    ]
    _write("tabla_resumen.tex", latex_table(
        ["Métrica", "C++ (Eigen)", "scikit-learn"],
        [("Calidad", quality), (f"Tiempo de entrenamiento (ms, $n={t['cpp']['n']}$ por lado)", time_rows),
         (f"Memoria (mediana y mín--máx, {b['method']['memory_runs_per_side']} procesos)", mem_rows)],
        "lrr"))

    # per-class test metrics (identical for both sides: enforced by the pipeline)
    cm_cpp, cm_sk = plots._class_metrics(d["cpp_pred"]), plots._class_metrics(d["sk_pred"])
    if not np.array_equal(cm_cpp.to_numpy(), cm_sk.to_numpy()):
        raise RuntimeError("per-class metrics differ between C++ and scikit-learn")
    support = d["sk_pred"]["true"].value_counts().sort_index()
    rows = [[tex(cls), fixed(cm_sk.loc[cls, "precision"], 4), fixed(cm_sk.loc[cls, "recall"], 4),
             fixed(cm_sk.loc[cls, "f1"], 4), str(int(support[i]))]
            for i, cls in enumerate(plots.CLASS_NAMES)]
    _write("tabla_clases.tex", latex_table(
        ["Clase", "Precisión", "Exhaustividad", "F1", "Soporte"], [(None, rows)], "lrrrr"))

    # equivalence configuration, the sklearn column read from the real model object
    p = sklearn_model.build_model().get_params()
    code = lambda x: r"\texttt{" + tex(x) + "}"  # noqa: E731
    cfg_rows = [
        ["Arquitectura", f"{config.N_FEATURES}--{config.N_HIDDEN}--{config.N_CLASSES}",
         code(f"hidden_layer_sizes={p['hidden_layer_sizes']}")],
        ["Activación oculta", "ReLU", code(f"activation={p['activation']!r}")],
        ["Salida y pérdida", "softmax + entropía cruzada media", "softmax + log-loss (implícitos)"],
        ["Optimizador", "descenso de gradiente por lote completo", code(f"solver={p['solver']!r}")],
        ["Tamaño de lote", str(config.N_TRAIN), code(f"batch_size={p['batch_size']}")],
        ["Tasa de aprendizaje", macros["learningRate"],
         code(f"learning_rate_init={p['learning_rate_init']}") + ", "
         + code(f"learning_rate={p['learning_rate']!r}")],
        ["Momentum", "no", code(f"momentum={p['momentum']}") + ", "
         + code(f"nesterovs_momentum={p['nesterovs_momentum']}")],
        ["Regularización $L_2$", "no", code(f"alpha={p['alpha']}")],
        ["Orden de las muestras", "fijo", code(f"shuffle={p['shuffle']}")],
        ["Criterios de parada", "ninguno", code(f"tol={p['tol']}") + ", "
         + code(f"early_stopping={p['early_stopping']}") + ", "
         + code(f"n_iter_no_change={p['n_iter_no_change']}")],
        ["Pesos iniciales", "CSV compartido", "CSV compartido, copiados in situ a coefs\\_"],
        ["Épocas", macros["nEpochs"], code("max_iter") + r" / llamadas a \texttt{partial\_fit}"],
        ["Precisión numérica", r"\texttt{double}", r"\texttt{float64}"],
    ]
    _write("tabla_config.tex", latex_table(
        ["Aspecto", "C++ (Eigen)", "scikit-learn"], [(None, cfg_rows)], r"lp{4.1cm}p{5.6cm}"))

    # lines of code
    loc = loc_rows()
    total_cpp = sum(r[1] or 0 for r in loc)
    total_py = sum(r[2] or 0 for r in loc)
    loc_tex = [[label, "---" if a is None else str(a), "---" if c is None else str(c)]
               for label, a, c in loc]
    loc_tex.append([r"\textbf{Total}", rf"\textbf{{{total_cpp}}}", rf"\textbf{{{total_py}}}"])
    _write("tabla_loc.tex", latex_table(
        ["Componente", "C++ (líneas)", "Python (líneas)"], [(None, loc_tex)], "lrr"))
    _write("loc_macros.tex", "\n".join([
        rf"\newcommand{{\locCpp}}{{{total_cpp}\xspace}}",
        rf"\newcommand{{\locPython}}{{{total_py}\xspace}}",
        rf"\newcommand{{\locRatio}}{{{fixed(total_cpp / total_py, 1)}\xspace}}",
        rf"\newcommand{{\locCppCore}}{{{loc[0][1]}\xspace}}",
        rf"\newcommand{{\locPythonCore}}{{{loc[0][2]}\xspace}}"]) + "\n")


# ------------------------------------------------------------------ figures, zip, entry point

def copy_figures() -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for pdf in sorted((config.RESULTS_DIR / "figures").glob("*.pdf")):
        shutil.copy2(pdf, FIGURES_DIR / pdf.name)
        copied.append(FIGURES_DIR / pdf.name)
    return copied


def generate() -> list[Path]:
    d = plots.load_inputs(config.RESULTS_DIR)
    if d["bench"] is None:
        raise RuntimeError("results/benchmark.json not found: run the pipeline without --no-bench")
    history_path = config.RESULTS_DIR / "benchmark_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    if not history:
        history = [{"speedup_median": d["bench"]["time_s"]["speedup_median"]}]
    macros = build_macros(d, run_binary_checks(), history)
    write_numbers(macros)
    write_tables(d, macros)
    return copy_figures() + sorted(GENERATED_DIR.glob("*.tex"))


def make_zip() -> Path:
    """Zip the self-contained report folder (no build products) for Overleaf."""
    include = [REPORT_DIR / "main.tex", REPORT_DIR / "references.bib"]
    include += sorted(FIGURES_DIR.glob("*.pdf")) + sorted(GENERATED_DIR.glob("*.tex"))
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for path in include:
            z.write(path, path.relative_to(REPORT_DIR))
    return ZIP_PATH


if __name__ == "__main__":
    written = generate()
    print(f"wrote {len(written)} report files to {REPORT_DIR.relative_to(config.ROOT)}/")
    if "--zip" in sys.argv:
        print(f"zip: {make_zip()}")
