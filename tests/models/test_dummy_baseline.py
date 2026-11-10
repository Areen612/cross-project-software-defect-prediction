import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.models.dummy_baseline import train_dummy_fold
from src.models.dummy_lopo_runner import run_dummy_lopo_experiment


class SpyDummy:
    def fit(self, X, y):
        self.fit_X = X.copy()
        self.fit_y = y.copy()
        self.classes_ = np.unique(y)
        self.most_frequent_ = self.classes_[np.argmax(np.bincount(y.astype(int)))]
        return self

    def predict(self, X):
        return np.repeat(self.most_frequent_, len(X))


def _data():
    X = pd.DataFrame(
        {"loc": [1.0, 3.0, 5.0, 7.0, 100.0, 200.0],
         "cbo": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]},
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


def test_dummy_fold_uses_only_training_labels_and_returns_aligned_predictions():
    X, y, metadata = _data()
    result = train_dummy_fold(
        X, y, metadata, "C",
        np.array([0, 1, 2, 3]),
        np.array([4, 5]),
    )

    assert isinstance(result["model"], DummyClassifier)
    assert result["model"].strategy == "most_frequent"
    # The training labels are tied (two True and two False).
    # DummyClassifier resolves this tie by selecting the first sorted class.
    assert result["predictions"].tolist() == [False, False]
    pd.testing.assert_series_equal(result["y_test"], y.iloc[[4, 5]])
    pd.testing.assert_frame_equal(
        result["metadata_test"], metadata.iloc[[4, 5]]
    )
    assert len(result["predictions"]) == len(result["y_test"])


def test_dummy_fold_records_fold_local_imputer_and_dropped_features():
    X, y, metadata = _data()
    X["all_missing_in_train"] = [np.nan, np.nan, np.nan, np.nan, 5.0, 6.0]

    result = train_dummy_fold(
        X, y, metadata, "C",
        np.array([0, 1, 2, 3]),
        np.array([4, 5]),
    )

    assert "all_missing_in_train" in result["dropped_features"]
    assert result["imputer"].statistics_[0] == 4.0


def test_dummy_lopo_runner_creates_one_fold_per_project():
    X, y, metadata = _data()
    results = run_dummy_lopo_experiment(X, y, metadata)

    assert len(results) == metadata["Project"].nunique()
    assert {result["held_out_project"] for result in results} == {"A", "B", "C"}

    predicted_indices = []
    for result in results:
        predicted_indices.extend(result["metadata_test"].index.tolist())
        assert result["metadata_test"]["Project"].nunique() == 1
        assert result["metadata_test"]["Project"].iloc[0] == result["held_out_project"]

    assert len(predicted_indices) == len(X)
    assert len(set(predicted_indices)) == len(X)



def test_dummy_classifier_learns_majority_class_from_training_labels():
    X, y, metadata = _data()
    # Training labels have two positives and two negatives, a tie.
    # This test is a training-label-only check; tie behavior is tested above.
    result = train_dummy_fold(
        X, y, metadata, "C",
        np.array([0, 1, 2, 3]),
        np.array([4, 5]),
    )
    assert result["predictions"].tolist() == [False, False]

    # Make the training split unbalanced: three positives, one negative.
    result = train_dummy_fold(
        X, y, metadata, "C",
        np.array([0, 2, 4, 1]),
        np.array([3, 5]),
    )
    assert result["predictions"].tolist() == [True, True]
