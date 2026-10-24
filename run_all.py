"""Orchestrate the full pipeline: data export, sklearn, C++ build/run, plots."""

import json
import sys

from python import compare, config, cpp_runner, data, sklearn_model, weights


def main() -> int:
    stages = 6

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
    if problems:
        print("MISMATCH between C++ and scikit-learn:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("C++ and scikit-learn are equivalent")

    stage(6, "generate plots and tables: not implemented yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
