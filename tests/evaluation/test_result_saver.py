from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from src.evaluation.result_saver import (
    build_predictions_dataframe,
    save_predictions,
    save_models,
    save_experiment_metadata
)

import pandas as pd
import numpy as np
import pytest
import joblib
import json


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
    
    
# --------------------------------------------------
# save_models tests
# --------------------------------------------------


@pytest.fixture
def sample_model_results(sample_results):
    """Add fitted, serializable models and imputers to fold results."""

    for result in sample_results:
        result["held_out_project"] = (
            result["metadata_test"]["Project"].iloc[0]
        )

        model = DummyClassifier(strategy="most_frequent")
        model.fit(
            np.array([[0], [1]]),
            np.array([False, True]),
        )

        imputer = SimpleImputer(strategy="median")
        imputer.fit(np.array([[1.0], [3.0]]))

        result["model"] = model
        result["imputer"] = imputer
        result["dropped_features"] = []

    sample_results[1]["dropped_features"] = ["cam"]

    return sample_results


def test_save_models_creates_models_directory(
    sample_model_results, tmp_path
):
    save_models(sample_model_results, tmp_path)

    assert (tmp_path / "models").is_dir()


def test_save_models_creates_one_file_per_fold(
    sample_model_results, tmp_path
):
    saved_paths = save_models(sample_model_results, tmp_path)

    assert len(saved_paths) == 2

    assert {path.name for path in saved_paths} == {
        "project_A.joblib",
        "project_B.joblib",
    }


def test_save_models_returns_saved_file_paths(
    sample_model_results, tmp_path
):
    saved_paths = save_models(sample_model_results, tmp_path)

    assert all(isinstance(path, Path) for path in saved_paths)
    assert all(path.exists() for path in saved_paths)


def test_save_models_accepts_string_output_directory(
    sample_model_results, tmp_path
):
    output_dir = str(tmp_path / "string_output")

    saved_paths = save_models(sample_model_results, output_dir)

    assert len(saved_paths) == 2
    assert all(path.exists() for path in saved_paths)


def test_save_models_accepts_path_output_directory(
    sample_model_results, tmp_path
):
    output_dir = tmp_path / "path_output"

    saved_paths = save_models(sample_model_results, output_dir)

    assert len(saved_paths) == 2
    assert all(path.exists() for path in saved_paths)


def test_save_models_saves_complete_model_bundle(
    sample_model_results, tmp_path
):
    saved_paths = save_models(sample_model_results, tmp_path)

    loaded_bundle = joblib.load(saved_paths[0])

    assert loaded_bundle["held_out_project"] == "project_A"
    assert isinstance(loaded_bundle["model"], DummyClassifier)
    assert isinstance(loaded_bundle["imputer"], SimpleImputer)
    assert loaded_bundle["dropped_features"] == []


def test_save_models_preserves_dropped_features(
    sample_model_results, tmp_path
):
    saved_paths = save_models(sample_model_results, tmp_path)

    project_b_path = next(
        path for path in saved_paths if path.stem == "project_B"
    )

    loaded_bundle = joblib.load(project_b_path)

    assert loaded_bundle["dropped_features"] == ["cam"]


def test_save_models_rejects_empty_results(tmp_path):
    with pytest.raises(
        ValueError,
        match="Experiment results cannot be empty",
    ):
        save_models([], tmp_path)


# --------------------------------------------------
# save_experiment_metadata tests
# --------------------------------------------------


@pytest.fixture
def sample_metadata():
    return {
        "experiment": {
            "dataset": "ds.csv",
            "number_of_observations": 110917,
            "number_of_projects": 39,
            "number_of_folds": 39,
        },
        "model": {
            "name": "RandomForestClassifier",
            "n_estimators": 200,
            "max_features": "sqrt",
            "random_state": 42,
        },
        "preprocessing": {
            "imputation_strategy": "median",
            "dropped_features": {
                "project_A": [],
                "project_B": ["cam"],
            },
        },
    }


def test_save_experiment_metadata_creates_json_file(
    sample_metadata, tmp_path
):
    saved_path = save_experiment_metadata(sample_metadata, tmp_path)

    assert saved_path.exists()
    assert saved_path.is_file()
    assert saved_path.name == "metadata.json"


def test_save_experiment_metadata_preserves_content(
    sample_metadata, tmp_path
):
    saved_path = save_experiment_metadata(sample_metadata, tmp_path)

    with saved_path.open("r", encoding="utf-8") as file:
        loaded_metadata = json.load(file)

    assert loaded_metadata == sample_metadata


def test_save_experiment_metadata_creates_missing_directory(
    sample_metadata, tmp_path
):
    output_dir = tmp_path / "nested" / "experiment"

    saved_path = save_experiment_metadata(sample_metadata, output_dir)

    assert output_dir.exists()
    assert saved_path.exists()


def test_save_experiment_metadata_returns_path(
    sample_metadata, tmp_path
):
    saved_path = save_experiment_metadata(sample_metadata, tmp_path)

    assert saved_path == tmp_path / "metadata.json"


def test_save_experiment_metadata_supports_empty_metadata(tmp_path):
    saved_path = save_experiment_metadata({}, tmp_path)

    with saved_path.open("r", encoding="utf-8") as file:
        loaded_metadata = json.load(file)

    assert loaded_metadata == {}