import re

from python import report_data

MAIN = report_data.REPORT_DIR / "main.tex"


def prose_without_math(source: str) -> str:
    body = source.split(r"\begin{document}", 1)[1]
    body = re.sub(r"(?m)(?<!\\)%.*$", "", body)  # comments
    body = re.sub(r"\\begin\{(align\*?|equation\*?)\}.*?\\end\{\1\}", "", body, flags=re.S)
    body = re.sub(r"\$[^$]*\$|\\\(.*?\\\)", "", body)  # inline math: numbers there are indices/exponents
    body = re.sub(r"[0-9.]+\\(linewidth|textwidth)|\\\\\[[0-9a-z]+\]", "", body)  # layout lengths
    body = re.sub(r"\\(cite|ref|label|input|lstinputlisting|includegraphics|"
                  r"bibliography|bibliographystyle)(\[[^\]]*\])?\{[^}]*\}", "", body)
    body = re.sub(r"\\texttt\{[^}]*\}", "", body)  # identifiers such as flags or code names
    body = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", body, flags=re.S)
    return body


def test_main_tex_has_no_hand_written_numbers():
    digits = re.findall(r".{0,25}\d+.{0,15}", prose_without_math(MAIN.read_text()))
    assert digits == [], "numbers must come from generated macros, found:\n" + "\n".join(digits)


def test_report_compiles_within_ten_pages(tmp_path):
    """Needs tectonic (a self-contained TeX engine); skipped when it is not installed."""
    import shutil
    import subprocess

    import pytest

    if shutil.which("tectonic") is None:
        pytest.skip("tectonic not installed")
    proc = subprocess.run(
        ["tectonic", "-X", "compile", "main.tex", "--outdir", str(tmp_path), "--keep-logs"],
        cwd=report_data.REPORT_DIR, capture_output=True, text=True)
    log = (tmp_path / "main.log").read_text() if (tmp_path / "main.log").exists() else ""
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "Undefined control sequence" not in log
    assert "LaTeX Warning: Reference" not in log or "undefined" not in log
    pages = int(re.search(r"\((\d+) pages?", log).group(1))
    assert pages <= 10, f"the report has {pages} pages (limit: 10)"
