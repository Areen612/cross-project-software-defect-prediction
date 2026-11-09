
import numpy as np
import pandas as pd
import sklearn
import time

from datetime import datetime
from pathlib import Path

from src.config import (
    RANDOM_SEED,
    N_ESTIMATORS,
    MAX_FEATURES,
    CLASS_WEIGHT,
    N_JOBS,
)
from src.data.data_loader import load_dataset
from src.data.target_builder import target_builder
from src.data.data_preparation import prepare_modeling_data
from src.models.lopo_runner import run_lopo_experiment
from src.evaluation.result_saver import (
    save_predictions,
    save_models,
    save_experiment_metadata,
)


def validate_experiment_results(results, X, y, metadata):
    """
    Validate the output of the complete LOPO experiment.

    Checks fold coverage, project separation, prediction alignment,
    and complete observation coverage.
    """

    # 1. Check the number of folds
    expected_projects = set(metadata["Project"].unique())

    assert len(results) == len(expected_projects), (
        "Incorrect number of folds."
    )

    # 2. Check every project was held out exactly once
    held_out_projects = [
        result["held_out_project"] for result in results
    ]

    assert set(held_out_projects) == expected_projects
    assert len(held_out_projects) == len(set(held_out_projects))

    tested_indices = []

    # 3. Validate each fold
    for result in results:
        project = result["held_out_project"]
        y_test = result["y_test"]
        predictions = result["predictions"]
        metadata_test = result["metadata_test"]

        # Every test observation must belong to the held-out project
        assert metadata_test["Project"].nunique() == 1
        assert metadata_test["Project"].iloc[0] == project

        # Actual labels and metadata must be aligned
        pd.testing.assert_index_equal(
            y_test.index,
            metadata_test.index,
        )

        # Every test observation must have one prediction
        assert len(predictions) == len(y_test)
        assert len(predictions) == len(metadata_test)

        # Predictions must be valid binary labels
        assert set(np.unique(predictions)).issubset({True, False})

        tested_indices.extend(metadata_test.index.tolist())

    # 4. Every observation must be tested exactly once
    assert len(tested_indices) == len(X)
    assert len(set(tested_indices)) == len(X)
    assert set(tested_indices) == set(X.index)

    # 5. Check all returned actual labels against original y
    for result in results:
        expected_y = y.loc[result["metadata_test"].index]

        pd.testing.assert_series_equal(
            result["y_test"],
            expected_y,
        )

    # 6. Check the number of predictions
    total_predictions = sum(
        len(result["predictions"]) for result in results
    )

    assert total_predictions == len(X)

    print("\nAll experiment validation checks passed!")


def main():
    # Generate a unique output directory for this experiment
    run_timestamp = datetime.now().astimezone()

    run_name = run_timestamp.strftime(
        "lopo_run_%Y%m%d_%H%M%S_%f"
    )

    output_dir = Path("outputs") / run_name

    # Load and prepare the dataset
    df = load_dataset()
    df = target_builder(df)

    X, y, metadata = prepare_modeling_data(df)

    print(f"Observations: {len(X):,}")
    print(f"Projects: {metadata['Project'].nunique()}")

    # Run the actual LOPO experiment
    print("\nStarting LOPO experiment...")
    
    start_time = time.perf_counter()
    results = run_lopo_experiment(X, y, metadata)
    elapsed_time = time.perf_counter() - start_time
    
    print(f"Experiment runtime: {elapsed_time:.2f} seconds")

    # Verify that every fold contains a fitted Random Forest
    for result in results:
        model = result["model"]

        print(
            f"{result['held_out_project']}: "
            f"{model.n_features_in_} features, "
            f"{len(result['y_test']):,} test observations"
        )

        assert model.n_features_in_ == (
            len(X.columns) - len(result["dropped_features"])
        )

        assert hasattr(model, "estimators_")
        assert len(model.estimators_) == N_ESTIMATORS

    print(
        f"All {len(results)} Random Forest models "
        f"contain {N_ESTIMATORS} fitted trees."
    )

    print("\nLOPO experiment completed.")

    # Validate the complete experiment output
    validate_experiment_results(results, X, y, metadata)

    total_predictions = sum(
        len(result["predictions"]) for result in results
    )

    print(f"\nTotal folds: {len(results)}")
    print(f"Total predictions: {total_predictions:,}")

    # Save predictions
    print("\nSaving experiment results...")

    predictions_path = save_predictions(
        results,
        output_dir,
    )

    print(f"Predictions saved: {predictions_path}")

    # Save fitted models and imputers
    model_paths = save_models(
        results,
        output_dir,
    )

    print(f"Saved models: {len(model_paths)}")

    # Collect experiment metadata
    experiment_metadata = {
        "experiment": {
            "name": "Leave-One-Project-Out",
            "timestamp": run_timestamp.isoformat(),
            "dataset": "ds.csv",
            "number_of_observations": len(X),
            "number_of_projects": metadata["Project"].nunique(),
            "number_of_versions": metadata["Version"].nunique(),
            "number_of_folds": len(results),
            "number_of_predictions": total_predictions,
            "runtime_seconds": round(elapsed_time, 2),
            "environment": {
                "scikit_learn_version": sklearn.__version__,
            }
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
            "name": "RandomForestClassifier",
            "n_estimators": N_ESTIMATORS,
            "max_features": MAX_FEATURES,
            "class_weight": CLASS_WEIGHT,
            "random_state": RANDOM_SEED,
            "n_jobs": N_JOBS,
        },
        "preprocessing": {
            "imputation_strategy": "median",
            "dropped_features_by_fold": {
                result["held_out_project"]: result["dropped_features"]
                for result in results
            },
        },
    }

    # Save experiment metadata
    metadata_path = save_experiment_metadata(
        experiment_metadata,
        output_dir,
    )

    print(f"Metadata saved: {metadata_path}")

    print("\nAll experiment artifacts saved successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()