import pytest

from python import config, plots

pytestmark = pytest.mark.skipif(
    not (config.RESULTS_DIR / "benchmark.json").exists(),
    reason="run `uv run run_all.py` first to produce the results",
)


def test_generate_all_writes_figures_and_tables(tmp_path):
    for name in ("sklearn_results.json", "cpp_results.json", "benchmark.json"):
        assert (config.RESULTS_DIR / name).exists(), name
    out = tmp_path / "figures"
    written = plots.generate_all(config.RESULTS_DIR, out)
    names = {p.name for p in written}
    for fig in ("fig_loss_curves", "fig_confusion_matrix", "fig_time", "fig_memory"):
        assert f"{fig}.png" in names and f"{fig}.pdf" in names
    assert {"summary_table.csv", "summary_table.md"} <= names


def test_table_has_per_class_metrics_and_notes():
    d = plots.load_inputs(config.RESULTS_DIR)
    rows, notes = plots.build_table(d)
    metrics = [m for _, m, _, _ in rows]
    for cls in plots.CLASS_NAMES:
        assert {f"{cls} precision", f"{cls} recall", f"{cls} f1"} <= set(metrics)
    assert any("warm_start=True" in n for n in notes)
    assert any("pandas" in n for n in notes)
    # no pipe characters inside a metric name (they would break the markdown table)
    assert not any("|" in m for m in metrics)
