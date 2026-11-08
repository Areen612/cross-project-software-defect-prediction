
import numpy as np
import pandas as pd
import pytest

from src.models.lopo_runner import run_lopo_experiment


class LightweightModel:
    """A lightweight model for testing the complete LOPO pipeline."""

    def fit(self, X, y):
        self.X_fit = X.copy()
        self.y_fit = y.copy()
        return self

    def predict(self, X):
        self.X_predict = X.copy()
        return np.zeros(len(X), dtype=bool)


@pytest.fixture
def synthetic_data():
    # Three projects, each containing two observations.
    # Non-default indices also help verify positional indexing.
    X = pd.DataFrame(
        {
            "loc": [10, 20, 30, 40, 50, 60],
            "cbo": [2, 4, 6, 8, 10, 12],
            "rfc": [5, np.nan, 15, 20, 25, 30],
        },
        index=[10, 20, 30, 40, 50, 60],
    )

    y = pd.Series(
        [True, False, True, False, True, False],
        index=X.index,
        name="buggy",
    )

    metadata = pd.DataFrame(
        {
            "Project": ["A", "A", "B", "B", "C", "C"],
            "Version": ["1", "1", "2", "2", "3", "3"],
            "Class": ["A1", "A2", "B1", "B2", "C1", "C2"],
        },
        index=X.index,
    )

    return X, y, metadata


def test_complete_lopo_pipeline(
    synthetic_data, monkeypatch
):
    X, y, metadata = synthetic_data

    created_models = []

    def create_lightweight_model():
        model = LightweightModel()
        created_models.append(model)
        return model

    # Replace only the Random Forest factory.
    # The real splitter, imputer, fold trainer, and runner remain active.
    monkeypatch.setattr(
        "src.models.fold_trainer.create_random_forest",
        create_lightweight_model,
    )

    results = run_lopo_experiment(X, y, metadata)

    # 1. Exactly one fold must be generated per project.
    expected_projects = set(metadata["Project"].unique())

    assert len(results) == len(expected_projects) == 3

    held_out_projects = [
        result["held_out_project"] for result in results
    ]

    assert set(held_out_projects) == expected_projects
    assert len(held_out_projects) == len(set(held_out_projects))

    # 2. Every fold must test only its designated project.
    for result in results:
        held_out_project = result["held_out_project"]
        metadata_test = result["metadata_test"]

        assert metadata_test["Project"].nunique() == 1
        assert metadata_test["Project"].iloc[0] == held_out_project

    # 3. Training and test projects must be disjoint.
    for model, result in zip(created_models, results):
        train_projects = set(
            metadata.loc[model.X_fit.index, "Project"]
        )

        test_projects = set(
            result["metadata_test"]["Project"]
        )

        assert train_projects.isdisjoint(test_projects)
        assert result["held_out_project"] not in train_projects

    # 4. Every observation must receive exactly one prediction.
    tested_indices = []

    for result in results:
        tested_indices.extend(result["metadata_test"].index.tolist())

    assert len(tested_indices) == len(metadata)
    assert len(set(tested_indices)) == len(metadata)
    assert set(tested_indices) == set(metadata.index)

    # 5. Actual labels, predictions, and metadata must stay aligned.
    for result in results:
        y_test = result["y_test"]
        predictions = result["predictions"]
        metadata_test = result["metadata_test"]

        pd.testing.assert_index_equal(
            y_test.index,
            metadata_test.index,
        )

        assert len(predictions) == len(y_test)
        assert len(predictions) == len(metadata_test)

    # 6. The model must fit on training observations
    #    and predict on the corresponding test observations.
    for model, result in zip(created_models, results):
        pd.testing.assert_index_equal(
            model.X_predict.index,
            result["metadata_test"].index,
        )

        assert len(model.X_fit) + len(model.X_predict) == len(X)

    # 7. Each fold must have its own model and fitted imputer.
    assert len({id(model) for model in created_models}) == 3

    imputers = [result["imputer"] for result in results]

    assert len({id(imputer) for imputer in imputers}) == 3
