"""Clean-process worker for the scikit-learn memory measurement.

Run as ``python -m python.bench_worker``. It prints one JSON line with the same fields as
the C++ ``--mem`` mode. Training increment = growth of the peak RSS (high-water mark)
during the single fit() call; imports and data loading happen before the baseline.
"""

import json
import resource
import sys

from . import config, data, sklearn_model


def peak_rss_bytes() -> int:
    """ru_maxrss is in BYTES on macOS but in KILOBYTES on Linux."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru if sys.platform == "darwin" else ru * 1024


def current_rss_bytes() -> int:
    import psutil

    return psutil.Process().memory_info().rss


def main() -> None:
    X_train, _, y_train, _ = data.load_csv()
    clf = sklearn_model.prepare_single_fit(X_train, y_train)
    rss_before, peak_before = current_rss_bytes(), peak_rss_bytes()
    sklearn_model.timed_fit(clf, X_train, y_train)
    peak_after = peak_rss_bytes()
    reference = json.loads((config.RESULTS_DIR / "sklearn_results.json").read_text())["train_loss"]
    assert abs(sklearn_model.cross_entropy(clf, X_train, y_train) - reference) < 1e-9
    print(json.dumps({
        "rss_before_train_bytes": rss_before,
        "peak_before_train_bytes": peak_before,
        "peak_after_train_bytes": peak_after,
        "train_increment_bytes": peak_after - peak_before,
    }))


if __name__ == "__main__":
    main()
