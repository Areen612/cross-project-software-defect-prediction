"""Generate robustness diagnostics from saved LOPO predictions."""
import argparse
from pathlib import Path

from src.evaluation.robustness import generate_robustness_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze saved Random Forest and Dummy LOPO predictions."
    )
    parser.add_argument("run_dir", type=Path,
                        help="RF run directory with predictions.csv and dummy_baseline/.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional output directory; defaults to <run_dir>/robustness.")
    args = parser.parse_args()
    output = generate_robustness_analysis(args.run_dir, args.output_dir)
    print(f"Robustness analysis completed: {output}")
    print(f"Project metrics: {output / 'project_robustness.csv'}")
    print(f"Version metrics: {output / 'version_robustness.csv'}")
    print(f"Summary: {output / 'robustness_summary.csv'}")


if __name__ == "__main__":
    main()
