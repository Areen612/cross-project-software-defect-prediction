"""Generate an evaluation workbook from an existing LOPO output directory."""

import argparse
from pathlib import Path

from src.evaluation.workbook_generator import generate_evaluation_workbook


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an Excel evaluation workbook from saved LOPO predictions."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Experiment directory containing predictions.csv and metadata.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .xlsx path. Defaults to <run_dir>/evaluation_workbook.xlsx.",
    )
    args = parser.parse_args()

    workbook_path = generate_evaluation_workbook(args.run_dir, args.output)
    print(f"Evaluation workbook saved: {workbook_path}")


if __name__ == "__main__":
    main()
