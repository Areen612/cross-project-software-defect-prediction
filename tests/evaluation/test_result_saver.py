import pandas as pd
import pytest

from src.evaluation.result_saver import build_predictions_dataframe


@pytest.fixture
def sample_results():
    """Create two small synthetic LOPO fold results."""

    first_fold_metadata = pd.DataFrame(
        {
            "Project": ["project_A", "project_A"],
            "Version": ["1.0", "1.0"],
            "Class": ["ClassA", "ClassB"],
        },
        index=[10, 11],
    )

    first_fold_actual = pd.Series(
        [True, False],
        index=[10, 11],
        name="buggy",
    )

    first_fold_predictions = [True, True]

    second_fold_metadata = pd.DataFrame(
        {
            "Project": ["project_B", "project_B"],
            "Version": ["2.0", "2.1"],
            "Class": ["ClassC", "ClassD"],
        },
        index=[20, 21],
    )

    second_fold_actual = pd.Series(
        [False, True],
        index=[20, 21],
        name="buggy",
    )

    second_fold_predictions = [False, True]

    return [
        {
            "metadata_test": first_fold_metadata,
            "y_test": first_fold_actual,
            "predictions": first_fold_predictions,
        },
        {
            "metadata_test": second_fold_metadata,
            "y_test": second_fold_actual,
            "predictions": second_fold_predictions,
        },
    ]


def test_build_predictions_dataframe_combines_all_folds(sample_results):
    result = build_predictions_dataframe(sample_results)

    assert len(result) == 4


def test_build_predictions_dataframe_preserves_metadata(sample_results):
    result = build_predictions_dataframe(sample_results)

    assert result["Project"].tolist() == [
        "project_A",
        "project_A",
        "project_B",
        "project_B",
    ]

    assert result["Version"].tolist() == ["1.0", "1.0", "2.0", "2.1"]

    assert result["Class"].tolist() == [
        "ClassA",
        "ClassB",
        "ClassC",
        "ClassD",
    ]


def test_build_predictions_dataframe_aligns_actual_and_predicted_labels(
    sample_results,
):
    result = build_predictions_dataframe(sample_results)

    assert result["Actual"].tolist() == [True, False, False, True]
    assert result["Predicted"].tolist() == [True, True, False, True]


def test_build_predictions_dataframe_returns_expected_columns(sample_results):
    result = build_predictions_dataframe(sample_results)

    expected_columns = [
        "Project",
        "Version",
        "Class",
        "Actual",
        "Predicted",
    ]

    assert result.columns.tolist() == expected_columns


def test_build_predictions_dataframe_resets_index(sample_results):
    result = build_predictions_dataframe(sample_results)

    assert result.index.tolist() == [0, 1, 2, 3]


def test_build_predictions_dataframe_rejects_empty_results():
    with pytest.raises(ValueError):
        build_predictions_dataframe([])