
import numpy as np
import pandas as pd
import pytest

from src.models.lopo_runner import run_lopo_experiment


@pytest.fixture
def synthetic_data():
    X = pd.DataFrame(
        {
            "loc": [100, 200, 300, 400],
            "cbo": [5, 10, 15, 20],
        }
    )

    y = pd.Series(
        [True, False, True, False],
        name="buggy",
    )

    metadata = pd.DataFrame(
        {
            "Project": ["A", "A", "B", "B"],
            "Version": ["1", "1", "2", "2"],
            "Class": ["A1", "A2", "B1", "B2"],
        }
    )

    return X, y, metadata


@pytest.fixture
def mock_splits():
    return [
        (
            "A",
            np.array([2, 3]),
            np.array([0, 1]),
        ),
        (
            "B",
            np.array([0, 1]),
            np.array([2, 3]),
        ),
    ]



def test_generates_splits_using_metadata(
    synthetic_data, mock_splits, monkeypatch
):
    X, y, metadata = synthetic_data

    received_arguments = []

    def mock_generate_splits(received_metadata):
        received_arguments.append(received_metadata)
        return mock_splits

    monkeypatch.setattr(
        "src.models.lopo_runner.generate_lopo_splits",
        mock_generate_splits,
    )

    monkeypatch.setattr(
        "src.models.lopo_runner.train_fold",
        lambda *args: {},
    )

    run_lopo_experiment(X, y, metadata)

    assert len(received_arguments) == 1
    assert received_arguments[0] is metadata

    # Verify that the supplied metadata was used to generate splits.
    # (The more explicit call assertion is covered below.)


def test_train_fold_called_once_per_split(
    synthetic_data, mock_splits, monkeypatch
):
    X, y, metadata = synthetic_data

    monkeypatch.setattr(
        "src.models.lopo_runner.generate_lopo_splits",
        lambda received_metadata: mock_splits,
    )

    train_fold_calls = []

    def mock_train_fold(*args):
        train_fold_calls.append(args)
        return {"held_out_project": args[3]}

    monkeypatch.setattr(
        "src.models.lopo_runner.train_fold",
        mock_train_fold,
    )

    run_lopo_experiment(X, y, metadata)

    assert len(train_fold_calls) == len(mock_splits)


def test_train_fold_receives_correct_arguments(
    synthetic_data, mock_splits, monkeypatch
):
    X, y, metadata = synthetic_data

    monkeypatch.setattr(
        "src.models.lopo_runner.generate_lopo_splits",
        lambda received_metadata: mock_splits,
    )

    train_fold_calls = []

    def mock_train_fold(*args):
        train_fold_calls.append(args)
        return {"held_out_project": args[3]}

    monkeypatch.setattr(
        "src.models.lopo_runner.train_fold",
        mock_train_fold,
    )

    run_lopo_experiment(X, y, metadata)

    for call, split in zip(train_fold_calls, mock_splits):
        held_out_project, train_indices, test_indices = split

        assert call[0] is X
        assert call[1] is y
        assert call[2] is metadata
        assert call[3] == held_out_project
        np.testing.assert_array_equal(call[4], train_indices)
        np.testing.assert_array_equal(call[5], test_indices)


def test_returns_all_fold_results_in_order(
    synthetic_data, mock_splits, monkeypatch
):
    X, y, metadata = synthetic_data

    monkeypatch.setattr(
        "src.models.lopo_runner.generate_lopo_splits",
        lambda received_metadata: mock_splits,
    )

    def mock_train_fold(
        X, y, metadata, held_out_project, train_indices, test_indices
    ):
        return {
            "held_out_project": held_out_project,
            "train_indices": train_indices,
            "test_indices": test_indices,
        }

    monkeypatch.setattr(
        "src.models.lopo_runner.train_fold",
        mock_train_fold,
    )

    results = run_lopo_experiment(X, y, metadata)

    assert isinstance(results, list)
    assert len(results) == len(mock_splits)

    assert [
        result["held_out_project"] for result in results
    ] == ["A", "B"]


def test_returns_empty_list_when_no_splits(
    synthetic_data, monkeypatch
):
    X, y, metadata = synthetic_data

    monkeypatch.setattr(
        "src.models.lopo_runner.generate_lopo_splits",
        lambda received_metadata: [],
    )

    train_fold_calls = []

    def mock_train_fold(*args):
        train_fold_calls.append(args)
        return {}

    monkeypatch.setattr(
        "src.models.lopo_runner.train_fold",
        mock_train_fold,
    )

    results = run_lopo_experiment(X, y, metadata)

    assert results == []
    assert train_fold_calls == []
