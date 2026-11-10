"""Run reproducible feature-ablation experiments against a saved RF reference run."""
import argparse
from pathlib import Path

from src.evaluation.feature_ablation import run_feature_ablation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run feature ablations with the project's LOPO Random Forest protocol."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Existing RF run directory used only to validate observation alignment.",
    )
    args = parser.parse_args()
    output = run_feature_ablation(args.run_dir)
    print(f"Feature ablation completed: {output}")
    print(f"Project metrics: {output / 'project_metrics_all.csv'}")
    print(f"Version metrics: {output / 'version_metrics_all.csv'}")
    print(f"Macro metrics: {output / 'macro_metrics_all.csv'}")


if __name__ == "__main__":
    main()
