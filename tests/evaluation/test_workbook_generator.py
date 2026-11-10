import json

import pandas as pd
import pytest
from openpyxl import load_workbook

from src.evaluation.workbook_generator import (
    SHEET_NAMES,
    generate_evaluation_workbook,
)


@pytest.fixture
def run_directory(tmp_path):
    run_dir = tmp_path / "lopo_run_test"
    run_dir.mkdir()

    predictions = pd.DataFrame(
        {
            "Project": ["project_A"] * 4 + ["project_B"] * 4,
            "Version": ["1.0", "1.0", "1.1", "1.1", "2.0", "2.0", "2.1", "2.1"],
            "Class": [f"Class{i}" for i in range(8)],
            "Actual": [True, False, True, False, False, True, False, True],
            "Predicted": [True, False, False, False, False, True, True, True],
        }
    )
    predictions.to_csv(run_dir / "predictions.csv", index=False)

    metadata = {
        "experiment": {
            "name": "Leave-One-Project-Out",
            "timestamp": "2026-11-10T10:00:00+03:00",
            "dataset": "ds.csv",
            "number_of_observations": 8,
            "number_of_projects": 2,
            "number_of_versions": 4,
            "number_of_folds": 2,
            "number_of_predictions": 8,
            "runtime_seconds": 1.25,
            "environment": {"scikit_learn_version": "1.9.1"},
        },
        "target": {"column": "buggy", "definition": "buggy = bugs > 0"},
        "features": {"number_of_features": 18, "predictors": ["cbo", "loc"]},
        "model": {
            "name": "RandomForestClassifier",
            "n_estimators": 200,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42,
        },
        "preprocessing": {"imputation_strategy": "median"},
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return run_dir


def test_generate_workbook_creates_expected_file_and_sheets(run_directory):
    output_path = generate_evaluation_workbook(run_directory)

    assert output_path == run_directory / "evaluation_workbook.xlsx"
    assert output_path.is_file()

    workbook = load_workbook(output_path, read_only=True)
    assert workbook.sheetnames == SHEET_NAMES
    workbook.close()


def test_version_metrics_sheet_contains_project_version_results(run_directory):
    output_path = generate_evaluation_workbook(run_directory)

    workbook = load_workbook(output_path, data_only=True, read_only=True)
    worksheet = workbook["Version Metrics"]

    headers = [cell.value for cell in next(worksheet.iter_rows(max_row=1))]
    assert headers == [
        "Project", "Version", "Support", "Buggy_Count", "Non_Buggy_Count",
        "Accuracy", "Precision", "Recall", "F1", "MCC", "MCC_Status",
        "TN", "FP", "FN", "TP",
    ]
    assert worksheet.max_row == 5  # header + four Project-Version groups
    workbook.close()


def test_project_summary_contains_aggregated_results(run_directory):
    output_path = generate_evaluation_workbook(run_directory)

    workbook = load_workbook(output_path, data_only=True, read_only=True)
    worksheet = workbook["Project Summary"]
    headers = [cell.value for cell in next(worksheet.iter_rows(max_row=1))]

    assert "Version_Count" in headers
    assert "Accuracy_Weighted" in headers
    assert "Pooled_MCC" in headers
    assert "Pooled_MCC_Status" in headers
    assert worksheet.max_row == 3  # header + two projects
    workbook.close()


def test_model_comparison_marks_dummy_baseline_pending(run_directory):
    output_path = generate_evaluation_workbook(run_directory)

    workbook = load_workbook(output_path, data_only=True, read_only=True)
    worksheet = workbook["Model Comparison"]
    rows = list(worksheet.iter_rows(values_only=True))
    headers = rows[0]
    model_index = headers.index("Model")
    status_index = headers.index("Status")

    dummy_rows = [row for row in rows[1:] if row[model_index] == "Dummy (most_frequent)"]
    assert len(dummy_rows) == 1
    assert dummy_rows[0][status_index] == "Pending"
    workbook.close()


def test_run_notes_include_model_dataset_and_methodology(run_directory):
    output_path = generate_evaluation_workbook(run_directory)

    workbook = load_workbook(output_path, data_only=True, read_only=True)
    rows = list(workbook["Run Notes"].iter_rows(values_only=True))
    notes = {row[0]: row[1] for row in rows[1:]}

    assert notes["Model"] == "RandomForestClassifier"
    assert notes["Dataset"] == "ds.csv"
    assert notes["Target definition"] == "buggy = bugs > 0"
    assert notes["Imputation"] == "median"
    assert "equal project weighting" in notes["Macro averaging"]
    assert notes["Dummy baseline"].startswith("Not yet run")
    workbook.close()


def test_generator_supports_custom_output_path(run_directory, tmp_path):
    output_path = tmp_path / "reports" / "rf_results.xlsx"

    result = generate_evaluation_workbook(run_directory, output_path)

    assert result == output_path
    assert output_path.is_file()


def test_generator_rejects_missing_predictions_file(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Predictions file not found"):
        generate_evaluation_workbook(run_dir)


def test_generator_rejects_missing_metadata_file(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "Project": ["A"],
            "Version": ["1"],
            "Actual": [True],
            "Predicted": [True],
        }
    ).to_csv(run_dir / "predictions.csv", index=False)

    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        generate_evaluation_workbook(run_dir)


def test_generator_rejects_empty_predictions(run_directory):
    pd.DataFrame(
        columns=["Project", "Version", "Actual", "Predicted"]
    ).to_csv(run_directory / "predictions.csv", index=False)

    with pytest.raises(ValueError, match="Predictions CSV is empty"):
        generate_evaluation_workbook(run_directory)


def test_generator_rejects_missing_prediction_columns(run_directory):
    pd.DataFrame(
        {"Project": ["A"], "Actual": [True], "Predicted": [True]}
    ).to_csv(run_directory / "predictions.csv", index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        generate_evaluation_workbook(run_directory)


def test_generator_rejects_invalid_boolean_labels(run_directory):
    pd.DataFrame(
        {
            "Project": ["A"],
            "Version": ["1"],
            "Actual": ["yes"],
            "Predicted": ["no"],
        }
    ).to_csv(run_directory / "predictions.csv", index=False)

    with pytest.raises(ValueError, match="must contain boolean labels"):
        generate_evaluation_workbook(run_directory)


def test_generator_retains_missing_versions(run_directory):
    predictions = pd.read_csv(run_directory / "predictions.csv")
    predictions.loc[0, "Version"] = None
    predictions.to_csv(run_directory / "predictions.csv", index=False)

    output_path = generate_evaluation_workbook(run_directory)
    workbook = load_workbook(output_path, data_only=True, read_only=True)
    rows = list(workbook["Version Metrics"].iter_rows(values_only=True))
    headers = rows[0]
    version_index = headers.index("Version")

    assert any(row[version_index] is None for row in rows[1:])
    workbook.close()


def test_generator_does_not_modify_source_predictions(run_directory):
    predictions_path = run_directory / "predictions.csv"
    original = predictions_path.read_bytes()

    generate_evaluation_workbook(run_directory)

    assert predictions_path.read_bytes() == original



def test_generator_includes_dummy_metrics_when_baseline_predictions_exist(run_directory):
    dummy_dir = run_directory / "dummy_baseline"
    dummy_dir.mkdir()

    dummy_predictions = pd.DataFrame(
        {
            "Project": ["project_A"] * 4 + ["project_B"] * 4,
            "Version": ["1.0", "1.0", "1.1", "1.1", "2.0", "2.0", "2.1", "2.1"],
            "Class": [f"Class{i}" for i in range(8)],
            "Actual": [True, False, True, False, False, True, False, True],
            "Predicted": [True] * 8,
        }
    )
    dummy_predictions.to_csv(dummy_dir / "predictions.csv", index=False)

    output_path = generate_evaluation_workbook(run_directory)
    workbook = load_workbook(output_path, data_only=True, read_only=True)

    rows = list(workbook["Model Comparison"].iter_rows(values_only=True))
    headers = rows[0]
    model_index = headers.index("Model")
    metric_index = headers.index("Metric")
    status_index = headers.index("Status")

    dummy_rows = [
        row for row in rows[1:]
        if row[model_index] == "Dummy (most_frequent)"
    ]
    assert len(dummy_rows) == 10
    assert all(row[status_index] != "Pending" for row in dummy_rows)
    assert any(row[metric_index] == "Accuracy" for row in dummy_rows)

    notes = list(workbook["Run Notes"].iter_rows(values_only=True))
    notes_dict = {row[0]: row[1] for row in notes[1:]}
    assert notes_dict["Dummy baseline"] == "Evaluated using the same LOPO protocol."
    workbook.close()
