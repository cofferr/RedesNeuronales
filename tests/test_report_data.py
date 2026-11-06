import pytest

from python import config, report_data


def test_tex_escapes_special_characters():
    assert report_data.tex("a_b & 50% #1") == r"a\_b \& 50\% \#1"


def test_sci_formats_as_math():
    assert report_data.sci(6.66e-16) == r"\ensuremath{6.7\times10^{-16}}"
    assert report_data.sci(0.0) == r"\ensuremath{0}"


def test_sloc_function_ignores_docstring_comments_and_blanks(tmp_path):
    src = tmp_path / "m.py"
    src.write_text('def f(x):\n    """doc\n    more doc"""\n\n    # comment\n    y = x + 1\n    return y\n')
    assert report_data.sloc_function(src, "f") == 3  # def line + two statements


def test_sloc_file_ignores_cpp_comments(tmp_path):
    src = tmp_path / "a.hpp"
    src.write_text("// header\n/* block\n   comment */\nint x;\n\nint y;\n")
    assert report_data.sloc_file(src) == 2


needs_results = pytest.mark.skipif(
    not (config.RESULTS_DIR / "benchmark.json").exists()
    or not (report_data.REPORT_DIR / "main.tex").exists(),
    reason="needs results/ (run run_all.py) and report/")


@needs_results
def test_generate_writes_macros_tables_and_figures():
    files = {p.name for p in report_data.generate()}
    assert {"numbers.tex", "tabla_resumen.tex", "tabla_clases.tex", "tabla_config.tex",
            "tabla_loc.tex", "loc_macros.tex", "fig_loss_curves.pdf"} <= files


@needs_results
def test_zip_contains_only_report_sources():
    import zipfile

    report_data.generate()
    with zipfile.ZipFile(report_data.make_zip()) as z:
        names = set(z.namelist())
    assert {"main.tex", "references.bib", "generated/numbers.tex",
            "figures/fig_time.pdf"} <= names
    assert not any(n.endswith((".aux", ".log", "main.pdf")) for n in names)
