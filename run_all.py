"""Orchestrate the full pipeline: data export, sklearn, C++ build/run, plots."""

from python import config, data, sklearn_model, weights

STAGES = [
    "export data",
    "train scikit-learn model",
    "build C++ project",
    "run C++ binary",
    "collect results",
    "generate plots and tables",
]


def main() -> None:
    for i, stage in enumerate(STAGES, start=1):
        print(f"[{i}/{len(STAGES)}] {stage}", end=": ")
        if stage == "export data":
            data.export_csv()
            config.export_config()
            weights.export_initial_weights()
            sklearn_model.export_initial_probabilities()
            print("done")
        elif stage == "train scikit-learn model":
            r = sklearn_model.run()
            print(f"test acc {r['test_accuracy']:.4f}, single fit {r['time_single_fit_s']:.3f}s")
        else:
            print("not implemented yet")


if __name__ == "__main__":
    main()
