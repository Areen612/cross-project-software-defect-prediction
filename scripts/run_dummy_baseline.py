"""Run the most-frequent Dummy baseline for an existing dataset and save artifacts."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import sklearn

from src.data.data_loader import load_dataset
from src.data.target_builder import target_builder
from src.data.data_preparation import prepare_modeling_data
from src.models.dummy_lopo_runner import run_dummy_lopo_experiment
from src.evaluation.result_saver import save_predictions, save_experiment_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DummyClassifier(strategy='most_frequent') with the same LOPO protocol."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Existing Random Forest run directory. Dummy artifacts are saved under its dummy_baseline/ subdirectory.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {output_dir}")
    if not (output_dir / "predictions.csv").is_file():
        raise FileNotFoundError(f"Random Forest predictions not found in: {output_dir}")

    df = target_builder(load_dataset())
    X, y, metadata = prepare_modeling_data(df)

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
        raise RuntimeError("Dummy LOPO folds do not cover every project exactly once.")

    total_predictions = sum(len(result["predictions"]) for result in results)
    if total_predictions != len(X):
        raise RuntimeError(
            f"Dummy prediction coverage mismatch: expected {len(X)}, got {total_predictions}."
        )

    dummy_dir = output_dir / "dummy_baseline"
    predictions_path = save_predictions(results, dummy_dir)

    metadata = {
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
            "environment": {"scikit_learn_version": sklearn.__version__},
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
            "note": "Same fold-local preprocessing protocol as Random Forest; predictor values are ignored by DummyClassifier.",
        },
    }
    save_experiment_metadata(metadata, dummy_dir)

    print(f"Dummy LOPO completed in {runtime_seconds:.2f} seconds.")
    print(f"Folds: {len(results)}")
    print(f"Predictions: {total_predictions:,}")
    print(f"Predictions saved: {predictions_path}")
    print(f"Metadata saved: {dummy_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
