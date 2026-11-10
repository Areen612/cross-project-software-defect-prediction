from __future__ import annotations

import pandas as pd
import pytest

from src.visualization.loaders import (
    ArtifactError,
    load_ablation_artifacts,
    load_dataset_frame,
    load_evaluation_artifacts,
    load_robustness_artifacts,
    evaluation_project_metrics,
)


def test_load_robustness_artifacts_validates_and_loads_all_levels(robustness_run):
    data = load_robustness_artifacts(robustness_run)
    assert len(data.projects) == 4
    assert len(data.versions) == 4
    assert len(data.summary) == 14
    assert data.projects["Model"].nunique() == 2
    assert data.projects["MCC"].isna().sum() >= 1


def test_load_robustness_artifacts_reports_missing_source(robustness_run):
    (robustness_run / "robustness" / "version_robustness.csv").unlink()
    with pytest.raises(ArtifactError, match="missing source file"):
        load_robustness_artifacts(robustness_run)


def test_load_robustness_artifacts_rejects_missing_columns(robustness_run):
    path = robustness_run / "robustness" / "project_robustness.csv"
    pd.DataFrame({"Project": ["A"]}).to_csv(path, index=False)
    with pytest.raises(ArtifactError, match="missing required columns"):
        load_robustness_artifacts(robustness_run)


def test_load_robustness_artifacts_rejects_invalid_numeric_values(robustness_run):
    path = robustness_run / "robustness" / "project_robustness.csv"
    frame = pd.read_csv(path)
    frame["Accuracy"] = frame["Accuracy"].astype(object)
    frame.loc[0, "Accuracy"] = "high"
    frame.to_csv(path, index=False)
    with pytest.raises(ArtifactError, match="non-numeric"):
        load_robustness_artifacts(robustness_run)


def test_load_evaluation_workbook_normalizes_unavailable_metrics(robustness_run):
    artifacts = load_evaluation_artifacts(robustness_run)
    projects = evaluation_project_metrics(artifacts)
    unavailable = artifacts.comparison[
        artifacts.comparison["Status"].ne("Available")
    ]
    assert set(projects["Model"]) == {"RandomForest", "DummyClassifier"}
    assert unavailable["Value"].isna().all()
    assert set(artifacts.comparison["Evaluation_Level"]) == {"Project", "Macro average"}


def test_load_evaluation_workbook_rejects_missing_columns(robustness_run):
    path = robustness_run / "evaluation_workbook.xlsx"
    pd.DataFrame({"Model": ["Random Forest"]}).to_excel(
        path, sheet_name="Model Comparison", index=False
    )
    with pytest.raises(ArtifactError, match="missing required columns"):
        load_evaluation_artifacts(robustness_run)


def test_load_ablation_artifacts_uses_manifest_configurations(ablation_run):
    data = load_ablation_artifacts(ablation_run)
    assert list(data.metadata["configurations"]) == ["baseline_all_features", "without_loc"]
    assert set(data.macro["Configuration"]) == set(data.metadata["configurations"])


def test_load_ablation_artifacts_rejects_configuration_mismatch(ablation_run):
    path = ablation_run / "feature_ablation" / "macro_metrics_all.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.index[0], "Configuration"] = "invented_configuration"
    frame.to_csv(path, index=False)
    with pytest.raises(ArtifactError, match="configurations do not match"):
        load_ablation_artifacts(ablation_run)


def test_dataset_reader_uses_documented_dash_as_missing(dataset_csv):
    original = dataset_csv.read_bytes()
    frame = load_dataset_frame(dataset_csv)
    assert pd.isna(frame.loc[1, "loc"])
    assert dataset_csv.read_bytes() == original


def test_dataset_reader_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ArtifactError, match="no observations"):
        load_dataset_frame(path)
