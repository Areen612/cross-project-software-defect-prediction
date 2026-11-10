"""Read and validate saved experiment artifacts for visualization only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.visualization.config import WORKBOOK_MODEL_KEYS


class ArtifactError(ValueError):
    """Raised when a visualization source artifact is missing or invalid."""


@dataclass(frozen=True)
class RobustnessArtifacts:
    projects: pd.DataFrame
    versions: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class EvaluationArtifacts:
    comparison: pd.DataFrame


@dataclass(frozen=True)
class AblationArtifacts:
    metadata: dict
    projects: pd.DataFrame
    versions: pd.DataFrame
    macro: pd.DataFrame


def _read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise ArtifactError(f"missing source file: {path}")
    try:
        frame = pd.read_csv(path, low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ArtifactError(f"could not read {path}: {exc}") from exc
    if frame.empty:
        raise ArtifactError(f"source file has no data rows: {path}")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ArtifactError(f"{path} is missing required columns: {missing}")
    return frame


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ArtifactError(f"missing source file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactError(f"expected a JSON object in {path}")
    return data


def load_run_metadata(experiment_dir: str | Path) -> dict:
    """Load the main experiment manifest without touching model artifacts."""
    metadata = _read_json(Path(experiment_dir) / "metadata.json")
    if not isinstance(metadata.get("experiment"), dict):
        raise ArtifactError("run metadata is missing its experiment section")
    return metadata


def load_evaluation_artifacts(experiment_dir: str | Path) -> EvaluationArtifacts:
    """Load the workbook's primary, project, pooled, and weighted comparisons."""
    path = Path(experiment_dir) / "evaluation_workbook.xlsx"
    if not path.is_file():
        raise ArtifactError(f"missing source file: {path}")
    required = {
        "Model", "Evaluation_Level", "Project", "Metric", "Value", "Status", "Notes"
    }
    try:
        comparison = pd.read_excel(path, sheet_name="Model Comparison")
    except (OSError, ImportError, ValueError, KeyError) as exc:
        raise ArtifactError(f"could not read Model Comparison from {path}: {exc}") from exc
    if comparison.empty:
        raise ArtifactError(f"Model Comparison sheet has no data rows: {path}")
    missing = sorted(required - set(comparison.columns))
    if missing:
        raise ArtifactError(f"{path} is missing required columns: {missing}")
    comparison = _numeric(comparison, ["Value"], path)
    expected_models = {"Random Forest", "Dummy (most_frequent)"}
    found_models = set(comparison["Model"].dropna().astype(str))
    if expected_models - found_models:
        raise ArtifactError(
            f"{path} is missing model rows: {sorted(expected_models - found_models)}"
        )
    if (
        comparison[["Model", "Evaluation_Level", "Metric", "Status"]]
        .isna()
        .any()
        .any()
    ):
        raise ArtifactError(f"{path} has missing model, level, metric, or status values")
    project_rows = comparison["Evaluation_Level"].eq("Project")
    macro_rows = comparison["Evaluation_Level"].eq("Macro average")
    if comparison.loc[project_rows, "Project"].isna().any():
        raise ArtifactError(f"{path} has project-level rows without project names")
    if not project_rows.any() or not macro_rows.any():
        raise ArtifactError(f"{path} must contain project and macro-average rows")
    keys = ["Model", "Evaluation_Level", "Project", "Metric"]
    if comparison.duplicated(keys).any():
        raise ArtifactError(f"{path} has duplicate comparison rows for {keys}")
    comparison = comparison.copy()
    # The workbook stores zero placeholders for some undefined MCC entries;
    # Status is authoritative for visualization and those values remain missing.
    comparison.loc[comparison["Status"].ne("Available"), "Value"] = float("nan")
    mcc_rows = comparison["Metric"].isin(["MCC", "Pooled MCC"])
    _bounded(comparison.loc[mcc_rows], ["Value"], -1, 1, path)
    _bounded(comparison.loc[~mcc_rows & comparison["Value"].notna()], ["Value"], 0, 1, path)
    return EvaluationArtifacts(comparison=comparison)


def evaluation_project_metrics(artifacts: EvaluationArtifacts) -> pd.DataFrame:
    """Return the workbook's primary project metrics with internal model keys."""
    frame = artifacts.comparison[
        artifacts.comparison["Evaluation_Level"].eq("Project")
    ]
    metrics = frame.pivot(index=["Model", "Project"], columns="Metric", values="Value")
    metrics = metrics.reset_index()
    metrics["Model"] = metrics["Model"].map(WORKBOOK_MODEL_KEYS).fillna(metrics["Model"])
    return metrics


def _numeric(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            continue
        original = result[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            examples = original[invalid].astype(str).head(3).tolist()
            raise ArtifactError(
                f"{path} has non-numeric values in {column}: {examples}"
            )
        result[column] = converted
    return result


def _bounded(
    frame: pd.DataFrame,
    columns: Iterable[str],
    lower: float,
    upper: float,
    path: Path,
) -> None:
    for column in columns:
        values = frame[column].dropna()
        if ((values < lower) | (values > upper)).any():
            raise ArtifactError(
                f"{path} has values outside [{lower}, {upper}] in {column}"
            )


def _validate_model_names(frame: pd.DataFrame, path: Path) -> None:
    expected = {"RandomForest", "DummyClassifier"}
    found = set(frame["Model"].dropna().astype(str))
    missing = expected - found
    if missing:
        raise ArtifactError(f"{path} is missing model rows: {sorted(missing)}")


def load_robustness_artifacts(experiment_dir: str | Path) -> RobustnessArtifacts:
    """Load the saved project, version, and project-summary CSV files."""
    root = Path(experiment_dir) / "robustness"
    project_path = root / "project_robustness.csv"
    version_path = root / "version_robustness.csv"
    summary_path = root / "robustness_summary.csv"
    project_columns = {
        "Project", "Model", "Support", "Accuracy", "Precision", "Recall",
        "Specificity", "Balanced_Accuracy", "F1", "MCC",
        "Actual_Class_Status",
    }
    version_columns = project_columns | {"Version"}
    summary_columns = {
        "Model", "Metric", "Projects_Total", "Projects_With_Both_Classes",
        "Projects_Contributing", "Mean", "Median", "Std", "Min", "Q1",
        "Q3", "Max",
    }
    projects = _read_csv(project_path, project_columns)
    versions = _read_csv(version_path, version_columns)
    summary = _read_csv(summary_path, summary_columns)

    metric_columns = [
        "Accuracy", "Precision", "Recall", "Specificity",
        "Balanced_Accuracy", "F1", "MCC",
    ]
    projects = _numeric(projects, ["Support", *metric_columns], project_path)
    versions = _numeric(versions, ["Support", *metric_columns], version_path)
    summary = _numeric(
        summary,
        ["Projects_Total", "Projects_With_Both_Classes", "Projects_Contributing",
         "Mean", "Median", "Std", "Min", "Q1", "Q3", "Max"],
        summary_path,
    )
    for frame, path, keys in (
        (projects, project_path, ["Project", "Model"]),
        (versions, version_path, ["Project", "Version", "Model"]),
        (summary, summary_path, ["Model", "Metric"]),
    ):
        if frame.duplicated(keys).any():
            raise ArtifactError(f"{path} has duplicate rows for {keys}")
        _validate_model_names(frame, path)
    rates = [column for column in metric_columns if column != "MCC"]
    _bounded(projects, rates, 0, 1, project_path)
    _bounded(projects, ["MCC"], -1, 1, project_path)
    _bounded(versions, rates, 0, 1, version_path)
    _bounded(versions, ["MCC"], -1, 1, version_path)
    _bounded(summary, ["Mean", "Median", "Min", "Q1", "Q3", "Max"], -1, 1, summary_path)
    for metric in ("Accuracy", "Precision", "Recall", "Specificity", "Balanced_Accuracy", "F1"):
        _bounded(summary[summary["Metric"].eq(metric)], ["Mean", "Median", "Min", "Q1", "Q3", "Max"], 0, 1, summary_path)
    for frame, path in ((projects, project_path), (versions, version_path)):
        if frame["Support"].isna().any() or (frame["Support"] <= 0).any():
            raise ArtifactError(f"{path} has missing or non-positive Support")
    return RobustnessArtifacts(projects, versions, summary)


def load_ablation_artifacts(experiment_dir: str | Path) -> AblationArtifacts:
    """Load ablation metrics and the manifest that defines each configuration."""
    root = Path(experiment_dir) / "feature_ablation"
    metadata_path = root / "metadata.json"
    metadata = _read_json(metadata_path)
    configurations = metadata.get("configurations")
    if not isinstance(configurations, dict) or not configurations:
        raise ArtifactError(f"{metadata_path} has no configuration manifest")

    project_path = root / "project_metrics_all.csv"
    version_path = root / "version_metrics_all.csv"
    macro_path = root / "macro_metrics_all.csv"
    projects = _read_csv(
        project_path,
        {"Configuration", "Project", "Support", "F1", "MCC", "Pooled_MCC"},
    )
    versions = _read_csv(
        version_path,
        {"Configuration", "Project", "Version", "Support", "F1", "MCC"},
    )
    macro = _read_csv(
        macro_path,
        {
            "Configuration", "Project_Count", "Accuracy", "Precision", "Recall",
            "F1", "MCC", "Pooled_MCC", "MCC_Informative_Projects",
            "Pooled_MCC_Informative_Projects",
        },
    )
    score_columns = [
        column for column in
        ("Accuracy", "Precision", "Recall", "F1", "MCC", "Pooled_MCC")
        if column in macro.columns
    ]
    macro = _numeric(
        macro,
        ["Project_Count", "MCC_Informative_Projects",
         "Pooled_MCC_Informative_Projects", *score_columns],
        macro_path,
    )
    projects = _numeric(
        projects,
        ["Support", "Accuracy", "Precision", "Recall", "F1", "MCC", "Pooled_MCC"],
        project_path,
    )
    versions = _numeric(
        versions,
        ["Support", "Accuracy", "Precision", "Recall", "F1", "MCC"],
        version_path,
    )
    expected = set(configurations)
    for frame, path in ((projects, project_path), (versions, version_path), (macro, macro_path)):
        actual = set(frame["Configuration"].dropna().astype(str))
        if actual != expected:
            raise ArtifactError(
                f"{path} configurations do not match {metadata_path}: "
                f"expected {sorted(expected)}, found {sorted(actual)}"
            )
    if macro["Configuration"].duplicated().any():
        raise ArtifactError(f"{macro_path} must have one row per configuration")
    for frame, path, keys in (
        (projects, project_path, ["Configuration", "Project"]),
        (versions, version_path, ["Configuration", "Project", "Version"]),
    ):
        if frame.duplicated(keys).any():
            raise ArtifactError(f"{path} has duplicate rows for {keys}")
    rate_columns = [column for column in score_columns if column not in {"MCC", "Pooled_MCC"}]
    _bounded(macro, rate_columns, 0, 1, macro_path)
    _bounded(macro, [column for column in score_columns if column in {"MCC", "Pooled_MCC"}], -1, 1, macro_path)
    return AblationArtifacts(metadata, projects, versions, macro)


def load_dataset_frame(dataset_path: str | Path) -> pd.DataFrame:
    """Read the source CSV with the project's documented dash-as-missing rule."""
    path = Path(dataset_path)
    if not path.is_file():
        raise ArtifactError(f"dataset not found: {path}")
    try:
        frame = pd.read_csv(path, low_memory=False, na_values=["-"])
    except pd.errors.EmptyDataError as exc:
        raise ArtifactError(f"dataset has no observations: {path}") from exc
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ArtifactError(f"could not read dataset {path}: {exc}") from exc
    if frame.empty:
        raise ArtifactError(f"dataset has no observations: {path}")
    if frame.columns.duplicated().any():
        raise ArtifactError(f"dataset has duplicate column names: {path}")
    return frame
