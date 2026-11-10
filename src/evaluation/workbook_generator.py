"""Build an Excel evaluation workbook from saved LOPO experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from src.evaluation.metrics import (
    evaluate_predictions,
    aggregate_project_metrics,
    calculate_macro_average,
)


SHEET_NAMES = [
    "Version Metrics",
    "Project Summary",
    "Model Comparison",
    "Run Notes",
]

VERSION_COLUMNS = [
    "Project", "Version", "Support", "Buggy_Count", "Non_Buggy_Count",
    "Accuracy", "Precision", "Recall", "F1", "MCC", "MCC_Status",
    "TN", "FP", "FN", "TP",
]

PROJECT_COLUMNS = [
    "Project", "Version_Count", "Support", "Buggy_Count", "Non_Buggy_Count",
    "Accuracy", "Precision", "Recall", "F1", "MCC",
    "MCC_Informative_Versions",
    "Accuracy_Weighted", "Precision_Weighted", "Recall_Weighted",
    "F1_Weighted", "Pooled_MCC", "Pooled_MCC_Status",
    "TN", "FP", "FN", "TP",
]

COMPARISON_COLUMNS = [
    "Model", "Evaluation_Level", "Metric", "Value", "Status", "Notes",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read metadata JSON at '{path}': {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError("Experiment metadata must contain a JSON object.")
    return data


def _validate_predictions(predictions: pd.DataFrame) -> None:
    required = {"Project", "Version", "Actual", "Predicted"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Predictions CSV is missing required columns: {sorted(missing)}"
        )
    if predictions.empty:
        raise ValueError("Predictions CSV is empty.")

    for column in ("Actual", "Predicted"):
        if not pd.api.types.is_bool_dtype(predictions[column]):
            # CSV serialization normally produces bools, but accepting
            # unambiguous 0/1 values makes the input robust to CSV writers.
            values = set(predictions[column].dropna().unique())
            if (
                predictions[column].isna().any()
                or not values.issubset({0, 1, False, True})
            ):
                raise ValueError(
                    f"Column '{column}' must contain boolean labels."
                )
            predictions[column] = predictions[column].astype(bool)


def _build_model_comparison(macro_metrics: pd.DataFrame) -> pd.DataFrame:
    """Create RF comparison rows; Dummy is explicitly marked pending."""
    macro = macro_metrics.iloc[0]
    informative_mcc_count = int(macro["MCC_Informative_Projects"])
    informative_pooled_count = int(macro["Pooled_MCC_Informative_Projects"])

    rf_metrics = [
        ("Accuracy", macro["Accuracy"], "All projects", "Equal project weighting."),
        ("Precision", macro["Precision"], "All projects", "Equal project weighting."),
        ("Recall", macro["Recall"], "All projects", "Equal project weighting."),
        ("F1", macro["F1"], "All projects", "Equal project weighting."),
        (
            "MCC",
            macro["MCC"],
            f"{informative_mcc_count} informative projects",
            "Mean of project-level MCC; projects without informative version-level MCC are excluded.",
        ),
        (
            "Pooled MCC",
            macro["Pooled_MCC"],
            f"{informative_pooled_count} informative projects",
            "Mean of project-level pooled MCC; non-informative pooled results are excluded.",
        ),
        (
            "Support-weighted Accuracy",
            macro["Accuracy_Weighted"],
            "All projects",
            "Macro mean of project support-weighted version metrics.",
        ),
        (
            "Support-weighted Precision",
            macro["Precision_Weighted"],
            "All projects",
            "Macro mean of project support-weighted version metrics.",
        ),
        (
            "Support-weighted Recall",
            macro["Recall_Weighted"],
            "All projects",
            "Macro mean of project support-weighted version metrics.",
        ),
        (
            "Support-weighted F1",
            macro["F1_Weighted"],
            "All projects",
            "Macro mean of project support-weighted version metrics.",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for metric, value, contributing_groups, note in rf_metrics:
        rows.append({
            "Model": "Random Forest",
            "Evaluation_Level": "Macro average across projects",
            "Metric": metric,
            "Value": value,
            "Status": (
                "Available" if pd.notna(value)
                else "No informative groups"
            ),
            "Notes": f"{contributing_groups}. {note}",
        })

    rows.append({
        "Model": "Dummy (most_frequent)",
        "Evaluation_Level": "Macro average across projects",
        "Metric": "All comparison metrics",
        "Value": None,
        "Status": "Pending",
        "Notes": (
            "Run the DummyClassifier baseline and generate its evaluation "
            "before comparison."
        ),
    })
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def _build_run_notes(
    metadata: dict[str, Any],
    version_metrics: pd.DataFrame,
    project_metrics: pd.DataFrame,
    macro_metrics: pd.DataFrame,
) -> pd.DataFrame:
    experiment = metadata.get("experiment", {})
    model = metadata.get("model", {})
    target = metadata.get("target", {})
    features = metadata.get("features", {})
    preprocessing = metadata.get("preprocessing", {})

    notes = [
        ("Experiment", experiment.get("name", "Not recorded")),
        ("Timestamp", experiment.get("timestamp", "Not recorded")),
        ("Dataset", experiment.get("dataset", "Not recorded")),
        ("Observations in metadata", experiment.get("number_of_observations", "Not recorded")),
        ("Projects in metadata", experiment.get("number_of_projects", "Not recorded")),
        ("Versions in metadata", experiment.get("number_of_versions", "Not recorded")),
        ("LOPO folds", experiment.get("number_of_folds", "Not recorded")),
        ("Predictions in metadata", experiment.get("number_of_predictions", "Not recorded")),
        ("Runtime (seconds)", experiment.get("runtime_seconds", "Not recorded")),
        (
            "scikit-learn version",
            experiment.get("environment", {}).get(
                "scikit_learn_version", "Not recorded"
            ),
        ),
        ("Model", model.get("name", "Not recorded")),
        ("Number of estimators", model.get("n_estimators", "Not recorded")),
        ("Max features", model.get("max_features", "Not recorded")),
        ("Class weight", str(model.get("class_weight", "Not recorded"))),
        ("Random seed", model.get("random_state", "Not recorded")),
        ("Target column", target.get("column", "Not recorded")),
        ("Target definition", target.get("definition", "Not recorded")),
        ("Imputation", preprocessing.get("imputation_strategy", "Not recorded")),
        ("Number of predictors", features.get("number_of_features", "Not recorded")),
        ("Predictors", ", ".join(features.get("predictors", []))),
        ("Version groups evaluated", len(version_metrics)),
        ("Projects summarized", len(project_metrics)),
        ("Macro-average projects", int(macro_metrics.iloc[0]["Project_Count"])),
        (
            "Version-level aggregation",
            "Metrics are calculated separately for each Project-Version pair; missing versions are retained.",
        ),
        (
            "Project-level primary metrics",
            "Unweighted mean across versions; each version contributes equally.",
        ),
        (
            "Project-level supplementary metrics",
            "Support-weighted means across versions.",
        ),
        (
            "MCC handling",
            "MCC means exclude versions/projects without informative MCC; status columns distinguish non-informative and undefined cases.",
        ),
        (
            "Pooled MCC",
            "Calculated from summed confusion-matrix counts, not by averaging version-level MCC.",
        ),
        (
            "Macro averaging",
            "Each project contributes equally. MCC and pooled MCC include only informative projects.",
        ),
        (
            "Dummy baseline",
            "Not yet run; no baseline results are fabricated in this workbook.",
        ),
    ]
    return pd.DataFrame(notes, columns=["Item", "Value"])


def _format_worksheet(
    worksheet,
    dataframe: pd.DataFrame,
    table_name: str,
) -> None:
    worksheet.freeze_panes = "A2"
    worksheet.sheet_view.showGridLines = False
    worksheet.auto_filter.ref = worksheet.dimensions

    for cell in worksheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="243B53")
        cell.alignment = Alignment(horizontal="center", vertical="center")

    worksheet.row_dimensions[1].height = 28

    for column_cells in worksheet.columns:
        values = [
            len(str(cell.value)) if cell.value is not None else 0
            for cell in column_cells
        ]
        width = min(max(max(values, default=10) + 2, 12), 48)
        worksheet.column_dimensions[
            get_column_letter(column_cells[0].column)
        ].width = width

    if len(dataframe.columns) and len(dataframe) > 0:
        reference = (
            f"A1:{get_column_letter(len(dataframe.columns))}"
            f"{len(dataframe) + 1}"
        )
        table = Table(displayName=table_name, ref=reference)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        worksheet.add_table(table)

    numeric_formats = {
        "Accuracy", "Precision", "Recall", "F1", "MCC", "Pooled_MCC",
        "Accuracy_Weighted", "Precision_Weighted", "Recall_Weighted",
        "F1_Weighted", "Value",
    }
    for column_index, column_name in enumerate(dataframe.columns, start=1):
        if column_name in numeric_formats:
            for row_index in range(2, len(dataframe) + 2):
                worksheet.cell(
                    row=row_index,
                    column=column_index,
                ).number_format = "0.0000"


def generate_evaluation_workbook(
    run_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """
    Generate an Excel workbook from an existing run's predictions.csv and metadata.json.

    The Random Forest is evaluated from its saved predictions; no model rerun
    is performed. Dummy baseline entries remain marked Pending.
    """
    run_dir = Path(run_dir)
    predictions_path = run_dir / "predictions.csv"
    metadata_path = run_dir / "metadata.json"

    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_path}"
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    predictions = pd.read_csv(predictions_path, low_memory=False)
    _validate_predictions(predictions)
    metadata = _read_json(metadata_path)

    version_metrics = evaluate_predictions(predictions)
    project_metrics = aggregate_project_metrics(version_metrics)
    macro_metrics = calculate_macro_average(project_metrics)

    version_metrics = version_metrics.reindex(columns=VERSION_COLUMNS)
    project_metrics = project_metrics.reindex(columns=PROJECT_COLUMNS)
    comparison = _build_model_comparison(macro_metrics)
    run_notes = _build_run_notes(
        metadata,
        version_metrics,
        project_metrics,
        macro_metrics,
    )

    if output_path is None:
        output_path = run_dir / "evaluation_workbook.xlsx"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        sheets = [
            ("Version Metrics", version_metrics, "VersionMetrics"),
            ("Project Summary", project_metrics, "ProjectSummary"),
            ("Model Comparison", comparison, "ModelComparison"),
            ("Run Notes", run_notes, "RunNotes"),
        ]
        for sheet_name, dataframe, table_name in sheets:
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
            _format_worksheet(
                writer.book[sheet_name],
                dataframe,
                table_name,
            )

    return output_path
