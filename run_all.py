"""Orchestrate the full pipeline: data export, sklearn, C++ build/run, plots."""

import argparse
import json
import sys

from python import (bench, compare, config, cpp_runner, data, plots, report_data, sklearn_model,
                    weights)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-bench", action="store_true", help="skip the (slower) benchmarks")
    args = parser.parse_args()
    stages = 8

    def stage(i: int, name: str) -> None:
        print(f"[{i}/{stages}] {name}")

    stage(1, "export data")
    data.export_csv()
    config.export_config()
    weights.export_initial_weights()
    sklearn_model.export_initial_probabilities()
    sklearn_model.export_epoch_reference()

    stage(2, "train scikit-learn model")
    sklearn_model.run()

    try:
        stage(3, "build C++ project (Release)")
        cpp_runner.build()
        stage(4, "run C++ binary")
        cpp = cpp_runner.run()
    except cpp_runner.CppError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stage(5, "compare C++ vs scikit-learn")
    sk = json.loads((config.RESULTS_DIR / "sklearn_results.json").read_text())
    sk_curve, cpp_curve = compare.load_curves()
    print(compare.summary(sk, cpp, sk_curve, cpp_curve))
    problems = compare.find_mismatches(sk, cpp, sk_curve, cpp_curve)
    problems += compare.prediction_mismatches()
    if problems:
        print("MISMATCH between C++ and scikit-learn:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("C++ and scikit-learn are equivalent")

    stage(6, "benchmarks" + (" (skipped)" if args.no_bench else ""))
    if not args.no_bench:
        try:
            print(bench.summary(bench.run()))
        except (RuntimeError, cpp_runner.CppError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    stage(7, "generate plots and tables")
    written = plots.generate_all()
    print(f"wrote {len(written)} files to {config.RESULTS_DIR.relative_to(config.ROOT)}/ "
          f"(figures/, summary_table.csv, summary_table.md)")

    stage(8, "report data (macros, tables, figures) and Overleaf zip")
    if args.no_bench:
        print("skipped: the report needs fresh benchmark results")
    elif not (report_data.REPORT_DIR / "main.tex").exists():
        print("skipped: report/ is not part of this copy of the project")
    else:
        files = report_data.generate()
        print(f"wrote {len(files)} files to report/; zip: {report_data.make_zip().name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
