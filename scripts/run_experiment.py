
import numpy as np
import pandas as pd
import time

from src.data.data_loader import load_dataset
from src.data.target_builder import target_builder
from src.data.data_preparation import prepare_modeling_data
from src.models.lopo_runner import run_lopo_experiment


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
        assert len(model.estimators_) == 200    

    print("All 39 Random Forest models contain 200 fitted trees.")
    

    print("\nLOPO experiment completed.")

    # Validate the complete experiment output
    validate_experiment_results(results, X, y, metadata)

    print(f"\nTotal folds: {len(results)}")
    print(f"Total predictions: {sum(len(r['predictions']) for r in results):,}")


if __name__ == "__main__":
    main()
