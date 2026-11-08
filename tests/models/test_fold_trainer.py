
import numpy as np
import pandas as pd
import pytest

from src.models.fold_trainer import train_fold


class SpyModel:
    """A lightweight model that records the data passed to fit and predict."""

    def fit(self, X, y):
        self.X_fit = X.copy()
        self.y_fit = y.copy()
        return self

    def predict(self, X):
        self.X_predict = X.copy()
        return np.array([True, False])


@pytest.fixture
def synthetic_data():
    X = pd.DataFrame(
        {
            "loc": [1, 3, 5, 7, 1000, 2000],
            "cbo": [2, 4, 6, 8, 10, 12],
            "all_missing": [np.nan, np.nan, np.nan, np.nan, 10, 20],
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
            "Project": ["A", "A", "C", "C", "B", "B"],
            "Version": ["1", "1", "2", "2", "3", "3"],
            "Class": ["A1", "A2", "C1", "C2", "B1", "B2"],
        },
        index=X.index,
    )

    return X, y, metadata


@pytest.fixture
def fold_indices():
    train_indices = np.array([0, 1, 2, 3])
    test_indices = np.array([4, 5])

    return train_indices, test_indices


def test_train_fold_returns_expected_results(
    synthetic_data, fold_indices, monkeypatch
):
    X, y, metadata = synthetic_data
    train_indices, test_indices = fold_indices

    spy_model = SpyModel()

    monkeypatch.setattr(
        "src.models.fold_trainer.create_random_forest",
        lambda: spy_model,
    )

    results = train_fold(
        X, y, metadata, "B", train_indices, test_indices
    )

    expected_keys = {
        "held_out_project",
        "y_test",
        "predictions",
        "model",
        "imputer",
        "dropped_features",
        "metadata_test",
    }

    assert set(results.keys()) == expected_keys
    assert results["held_out_project"] == "B"
    assert results["model"] is spy_model
    assert results["imputer"] is not None


def test_train_fold_preserves_test_alignment(
    synthetic_data, fold_indices, monkeypatch
):
    X, y, metadata = synthetic_data
    train_indices, test_indices = fold_indices

    monkeypatch.setattr(
        "src.models.fold_trainer.create_random_forest",
        SpyModel,
    )

    results = train_fold(
        X, y, metadata, "B", train_indices, test_indices
    )

    pd.testing.assert_series_equal(
        results["y_test"],
        y.iloc[test_indices],
    )

    pd.testing.assert_frame_equal(
        results["metadata_test"],
        metadata.iloc[test_indices],
    )

    assert len(results["predictions"]) == len(results["y_test"])
    assert len(results["predictions"]) == len(results["metadata_test"])


def test_imputer_learns_medians_from_training_data_only(
    synthetic_data, fold_indices, monkeypatch
):
    X, y, metadata = synthetic_data
    train_indices, test_indices = fold_indices

    monkeypatch.setattr(
        "src.models.fold_trainer.create_random_forest",
        SpyModel,
    )

    results = train_fold(
        X, y, metadata, "B", train_indices, test_indices
    )

    # Median of training LOC values: [1, 3, 5, 7] = 4
    assert results["imputer"].statistics_[0] == 4

    # Test LOC values must not influence the fitted median.
    assert results["imputer"].statistics_[0] != 1000


def test_model_is_fitted_using_training_data_only(
    synthetic_data, fold_indices, monkeypatch
):
    X, y, metadata = synthetic_data
    train_indices, test_indices = fold_indices

    spy_model = SpyModel()

    monkeypatch.setattr(
        "src.models.fold_trainer.create_random_forest",
        lambda: spy_model,
    )

    train_fold(
        X, y, metadata, "B", train_indices, test_indices
    )

    # The model should receive only the four training observations.
    assert len(spy_model.X_fit) == len(train_indices)
    assert len(spy_model.y_fit) == len(train_indices)

    # The model should predict only on the two test observations.
    assert len(spy_model.X_predict) == len(test_indices)

    # Test observations must not appear in the training features.
    assert 1000 not in spy_model.X_fit["loc"].values
    assert 2000 not in spy_model.X_fit["loc"].values


def test_train_fold_reports_dropped_features(
    synthetic_data, fold_indices, monkeypatch
):
    X, y, metadata = synthetic_data
    train_indices, test_indices = fold_indices

    monkeypatch.setattr(
        "src.models.fold_trainer.create_random_forest",
        SpyModel,
    )

    results = train_fold(
        X, y, metadata, "B", train_indices, test_indices
    )

    assert "all_missing" in results["dropped_features"]


def test_train_fold_does_not_modify_original_data(
    synthetic_data, fold_indices, monkeypatch
):
    X, y, metadata = synthetic_data
    train_indices, test_indices = fold_indices

    X_original = X.copy(deep=True)
    y_original = y.copy(deep=True)
    metadata_original = metadata.copy(deep=True)

    monkeypatch.setattr(
        "src.models.fold_trainer.create_random_forest",
        SpyModel,
    )

    train_fold(
        X, y, metadata, "B", train_indices, test_indices
    )

    pd.testing.assert_frame_equal(X, X_original)
    pd.testing.assert_series_equal(y, y_original)
    pd.testing.assert_frame_equal(metadata, metadata_original)
