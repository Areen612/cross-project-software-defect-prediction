
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.result_saver import (
    build_predictions_dataframe,
    save_predictions,
)


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


# --------------------------------------------------
# build_predictions_dataframe tests
# --------------------------------------------------


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


# --------------------------------------------------
# save_predictions tests
# --------------------------------------------------


def test_save_predictions_creates_csv_file(sample_results, tmp_path):
    output_path = save_predictions(sample_results, tmp_path)

    assert output_path.exists()
    assert output_path.is_file()
    assert output_path.name == "predictions.csv"


def test_save_predictions_creates_nested_directories(sample_results, tmp_path):
    output_dir = tmp_path / "outputs" / "experiments" / "run_001"

    output_path = save_predictions(sample_results, output_dir)

    assert output_path.exists()
    assert output_dir.is_dir()


def test_save_predictions_accepts_string_output_directory(
    sample_results, tmp_path
):
    output_dir = str(tmp_path / "string_output")

    output_path = save_predictions(sample_results, output_dir)

    assert isinstance(output_path, Path)
    assert output_path.exists()


def test_save_predictions_accepts_path_output_directory(
    sample_results, tmp_path
):
    output_dir = tmp_path / "path_output"

    output_path = save_predictions(sample_results, output_dir)

    assert isinstance(output_path, Path)
    assert output_path.exists()


def test_save_predictions_saves_correct_dataframe(sample_results, tmp_path):
    output_path = save_predictions(sample_results, tmp_path)

    saved_dataframe = pd.read_csv(
        output_path,
        dtype={"Version": "string"},
    )

    expected_dataframe = build_predictions_dataframe(sample_results)

    pd.testing.assert_frame_equal(
        saved_dataframe,
        expected_dataframe,
        check_dtype=False,
    )


def test_save_predictions_does_not_save_dataframe_index(
    sample_results, tmp_path
):
    output_path = save_predictions(sample_results, tmp_path)

    saved_dataframe = pd.read_csv(output_path)

    assert saved_dataframe.columns.tolist() == [
        "Project",
        "Version",
        "Class",
        "Actual",
        "Predicted",
    ]


def test_save_predictions_supports_custom_filename(sample_results, tmp_path):
    output_path = save_predictions(
        sample_results,
        tmp_path,
        filename="custom_predictions.csv",
    )

    assert output_path.name == "custom_predictions.csv"
    assert output_path.exists()


def test_save_predictions_rejects_empty_results(tmp_path):
    with pytest.raises(
        ValueError,
        match="Experiment results cannot be empty",
    ):
        save_predictions([], tmp_path)


def test_save_predictions_can_use_existing_directory(
    sample_results, tmp_path
):
    output_dir = tmp_path / "existing_output"
    output_dir.mkdir()

    output_path = save_predictions(sample_results, output_dir)

    assert output_path.exists()