"""Orchestrate the full pipeline: data export, sklearn, C++ build/run, plots."""

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
        print(f"[{i}/{len(STAGES)}] {stage}: not implemented yet")


if __name__ == "__main__":
    main()
