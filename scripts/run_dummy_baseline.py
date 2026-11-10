"""Run the most-frequent Dummy baseline against an existing RF LOPO run."""

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import sklearn

from src.data.data_loader import load_dataset
from src.data.target_builder import target_builder
from src.data.data_preparation import prepare_modeling_data
from src.evaluation.prediction_alignment import validate_prediction_alignment
from src.evaluation.result_saver import (
    save_predictions,
    save_experiment_metadata,
)
from src.evaluation.workbook_generator import generate_evaluation_workbook
from src.models.dummy_lopo_runner import run_dummy_lopo_experiment


def _validate_rf_reference(
    run_dir: Path,
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
) -> None:
    """Ensure the currently loaded dataset matches the saved RF observations."""
    rf_predictions_path = run_dir / "predictions.csv"
    rf_metadata_path = run_dir / "metadata.json"

    if not rf_predictions_path.is_file():
        raise FileNotFoundError(
            f"Random Forest predictions not found in: {run_dir}"
        )
    if not rf_metadata_path.is_file():
        raise FileNotFoundError(
            f"Random Forest metadata not found in: {run_dir}"
        )

    rf_predictions = pd.read_csv(rf_predictions_path, low_memory=False)
    expected_observations = metadata[["Project", "Version", "Class"]].copy()
    expected_observations["Actual"] = y.to_numpy()

    if len(rf_predictions) != len(X):
        raise ValueError(
            "Current dataset observation count does not match the saved RF predictions."
        )

    validate_prediction_alignment(rf_predictions, expected_observations)


def run_baseline(output_dir: Path) -> Path:
    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist: {output_dir}"
        )

    dataset = target_builder(load_dataset())
    X, y, metadata = prepare_modeling_data(dataset)

    # Refuse to compare against a changed dataset or target definition.
    _validate_rf_reference(output_dir, X, y, metadata)

    started_at = time.perf_counter()
    results = run_dummy_lopo_experiment(X, y, metadata)
    runtime_seconds = round(time.perf_counter() - started_at, 2)

    expected_projects = set(metadata["Project"].unique())
    held_out_projects = [result["held_out_project"] for result in results]
    if (
        len(results) != len(expected_projects)
        or set(held_out_projects) != expected_projects
        or len(held_out_projects) != len(set(held_out_projects))
    ):
        raise RuntimeError(
            "Dummy LOPO folds do not cover every project exactly once."
        )

    total_predictions = sum(
        len(result["predictions"]) for result in results
    )
    if total_predictions != len(X):
        raise RuntimeError(
            f"Dummy prediction coverage mismatch: expected {len(X)}, "
            f"got {total_predictions}."
        )

    dummy_dir = output_dir / "dummy_baseline"
    predictions_path = save_predictions(results, dummy_dir)

    dummy_metadata = {
        "experiment": {
            "name": "Leave-One-Project-Out",
            "timestamp": datetime.now().astimezone().isoformat(),
            "dataset": "ds.csv",
            "number_of_observations": len(X),
            "number_of_projects": metadata["Project"].nunique(),
            "number_of_versions": metadata["Version"].nunique(),
            "number_of_folds": len(results),
            "number_of_predictions": total_predictions,
            "runtime_seconds": runtime_seconds,
            "environment": {
                "scikit_learn_version": sklearn.__version__,
            },
        },
        "target": {
            "column": "buggy",
            "definition": "buggy = bugs > 0",
        },
        "features": {
            "number_of_features": len(X.columns),
            "predictors": X.columns.tolist(),
        },
        "model": {
            "name": "DummyClassifier",
            "strategy": "most_frequent",
        },
        "preprocessing": {
            "imputation_strategy": "median",
            "note": (
                "Same fold-local preprocessing diagnostics as Random Forest. "
                "DummyClassifier chooses its prediction using training labels "
                "only and ignores predictor values."
            ),
        },
        "evaluation": {
            "split_protocol": "Leave-One-Project-Out",
            "reference_run": str(output_dir),
            "observation_alignment": (
                "Project, Version, Class, and Actual labels validated against "
                "the saved Random Forest predictions."
            ),
        },
    }
    metadata_path = save_experiment_metadata(dummy_metadata, dummy_dir)

    # Workbook generation performs a second alignment check before comparing.
    workbook_path = generate_evaluation_workbook(output_dir)

    print(f"Dummy LOPO completed in {runtime_seconds:.2f} seconds.")
    print(f"Folds: {len(results)}")
    print(f"Predictions: {total_predictions:,}")
    print(f"Predictions saved: {predictions_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Updated comparison workbook: {workbook_path}")
    return workbook_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run DummyClassifier(strategy='most_frequent') with the same "
            "LOPO protocol as an existing Random Forest run."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Existing Random Forest run directory containing predictions.csv "
            "and metadata.json. Dummy results are saved in dummy_baseline/."
        ),
    )
    args = parser.parse_args()
    run_baseline(args.output_dir)


if __name__ == "__main__":
    main()
