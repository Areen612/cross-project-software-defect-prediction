import numpy as np
import pandas as pd
import pytest

from src.evaluation.robustness import (
    analyze_prediction_robustness,
    generate_robustness_analysis,
    summarize_project_robustness,
)


def _sample_predictions():
    return pd.DataFrame({
        "Project": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"],
        "Version": ["1", "1", "2", "2", "1", "1", "2", "2", None, None],
        "Class": ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2"],
        "Actual": [True, False, True, False, True, False, True, False, True, True],
        "Predicted": [True, False, True, True, True, False, False, False, True, True],
    })


def test_project_metrics_pool_all_versions_within_project():
    projects, _, _ = analyze_prediction_robustness(_sample_predictions(), "RF")
    project_a = projects.set_index("Project").loc["A"]
    assert project_a["Support"] == 4
    assert project_a[["TN", "FP", "FN", "TP"]].tolist() == [1, 1, 0, 2]
    assert project_a["Accuracy"] == pytest.approx(0.75)


def test_version_metrics_retain_missing_versions():
    _, versions, _ = analyze_prediction_robustness(_sample_predictions(), "RF")
    assert len(versions) == 5
    missing_version = versions[versions["Version"].isna()]
    assert len(missing_version) == 1
    assert missing_version.iloc[0]["Support"] == 2


def test_single_class_project_has_undefined_specificity_and_balanced_accuracy():
    projects, _, summary = analyze_prediction_robustness(_sample_predictions(), "RF")
    project_c = projects.set_index("Project").loc["C"]
    assert project_c["Actual_Class_Status"] == "Single class"
    assert np.isnan(project_c["Specificity"])
    assert np.isnan(project_c["Balanced_Accuracy"])
    assert project_c["MCC_Status"] == "Not informative"
    row = summary[
        (summary["Model"] == "RF")
        & (summary["Metric"] == "Balanced_Accuracy")
    ].iloc[0]
    assert row["Aggregation_Level"] == "Project-pooled macro average"
    assert row["Projects_Total"] == 3
    assert row["Projects_With_Both_Classes"] == 2
    assert row["Projects_Contributing"] == 2


def test_single_class_predictions_make_mcc_undefined():
    predictions = _sample_predictions()
    predictions.loc[predictions["Project"] == "A", "Predicted"] = True
    projects, _, _ = analyze_prediction_robustness(predictions, "RF")
    project_a = projects.set_index("Project").loc["A"]
    assert project_a["MCC_Status"] == "Undefined"
    assert np.isnan(project_a["MCC"])


def test_project_summary_reports_distribution():
    projects, _, summary = analyze_prediction_robustness(_sample_predictions(), "RF")
    row = summary[summary["Metric"] == "Accuracy"].iloc[0]
    assert row["Mean"] == pytest.approx(projects["Accuracy"].mean())
    assert row["Median"] == pytest.approx(projects["Accuracy"].median())
    assert row["Min"] == pytest.approx(projects["Accuracy"].min())
    assert row["Max"] == pytest.approx(projects["Accuracy"].max())


@pytest.mark.parametrize("column", ["Project", "Version", "Actual", "Predicted"])
def test_rejects_missing_required_columns(column):
    with pytest.raises(ValueError, match="missing columns"):
        analyze_prediction_robustness(_sample_predictions().drop(columns=[column]), "RF")


def test_rejects_non_boolean_labels():
    predictions = _sample_predictions()
    predictions["Actual"] = predictions["Actual"].astype(int)
    with pytest.raises(ValueError, match="boolean dtype"):
        analyze_prediction_robustness(predictions, "RF")


def test_rejects_empty_predictions():
    with pytest.raises(ValueError, match="empty"):
        analyze_prediction_robustness(pd.DataFrame(), "RF")


def test_rejects_empty_model_name():
    with pytest.raises(ValueError, match="model_name"):
        analyze_prediction_robustness(_sample_predictions(), " ")


def test_generation_compares_rf_and_dummy_and_writes_outputs(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "dummy_baseline").mkdir(parents=True)
    predictions = _sample_predictions()
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    predictions.assign(Predicted=False).to_csv(
        run_dir / "dummy_baseline" / "predictions.csv", index=False
    )
    output = generate_robustness_analysis(run_dir)
    assert (output / "project_robustness.csv").is_file()
    assert (output / "version_robustness.csv").is_file()
    assert (output / "robustness_summary.csv").is_file()
    assert (output / "README.txt").is_file()
    notes = (output / "README.txt").read_text(encoding="utf-8")
    assert "equal-project macro average of project-pooled metrics" in notes
    result = pd.read_csv(output / "project_robustness.csv")
    assert set(result["Model"]) == {"RandomForest", "DummyClassifier"}
    assert len(result) == 6


def test_generation_rejects_misaligned_predictions(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "dummy_baseline").mkdir(parents=True)
    predictions = _sample_predictions()
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    changed = predictions.copy()
    changed.loc[0, "Actual"] = False
    changed.to_csv(run_dir / "dummy_baseline" / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="not aligned"):
        generate_robustness_analysis(run_dir)


def test_summary_rejects_empty_project_metrics():
    with pytest.raises(ValueError, match="empty"):
        summarize_project_robustness(pd.DataFrame())
