"""Project- and version-level robustness diagnostics for saved predictions.

The analysis uses saved out-of-project predictions only; it does not retrain models.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import matthews_corrcoef

from src.evaluation.prediction_alignment import validate_prediction_alignment

_REQUIRED = {"Project", "Version", "Actual", "Predicted"}
_METRICS = ["Accuracy", "Precision", "Recall", "Specificity",
            "Balanced_Accuracy", "F1", "MCC"]


def _validate_predictions(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("Prediction data is empty.")
    missing = _REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction data is missing columns: {sorted(missing)}")
    if frame[["Project", "Actual", "Predicted"]].isna().any().any():
        raise ValueError("Project, Actual, and Predicted cannot contain missing values.")
    for column in ("Actual", "Predicted"):
        if not pd.api.types.is_bool_dtype(frame[column]):
            raise ValueError(f"'{column}' must have boolean dtype.")


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def _group_metrics(frame: pd.DataFrame, keys: list[str], model: str) -> pd.DataFrame:
    rows = []
    for group_key, group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        actual, predicted = group["Actual"], group["Predicted"]
        tp = int((actual & predicted).sum())
        tn = int((~actual & ~predicted).sum())
        fp = int((~actual & predicted).sum())
        fn = int((actual & ~predicted).sum())
        positives, negatives = tp + fn, tn + fp
        predicted_positive, predicted_negative = tp + fp, tn + fn
        recall = _ratio(tp, positives)
        specificity = _ratio(tn, negatives)
        precision = _ratio(tp, predicted_positive)
        f1 = _ratio(2 * tp, 2 * tp + fp + fn)
        both_actual = positives > 0 and negatives > 0
        both_predicted = predicted_positive > 0 and predicted_negative > 0
        if not both_actual:
            mcc_status, mcc = "Not informative", math.nan
        elif not both_predicted:
            mcc_status, mcc = "Undefined", math.nan
        else:
            mcc_status, mcc = "Informative", float(matthews_corrcoef(actual, predicted))
        rows.append({
            **dict(zip(keys, group_key)),
            "Model": model,
            "Support": len(group),
            "Buggy_Count": positives,
            "Non_Buggy_Count": negatives,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "Accuracy": _ratio(tp + tn, len(group)),
            "Precision": 0.0 if math.isnan(precision) else precision,
            "Recall": recall,
            "Specificity": specificity,
            "Balanced_Accuracy": (recall + specificity) / 2 if both_actual else math.nan,
            "F1": 0.0 if math.isnan(f1) else f1,
            "MCC": mcc,
            "MCC_Status": mcc_status,
            "Actual_Class_Status": "Both classes" if both_actual else "Single class",
        })
    return pd.DataFrame(rows)


def summarize_project_robustness(project_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize project-to-project metric distributions; undefined values stay missing."""
    if project_metrics.empty:
        raise ValueError("Project metrics are empty.")
    rows = []
    for model, data in project_metrics.groupby("Model", sort=True):
        total = len(data)
        both_classes = int((data["Actual_Class_Status"] == "Both classes").sum())
        for metric in _METRICS:
            values = pd.to_numeric(data[metric], errors="coerce").dropna()
            rows.append({
                "Model": model,
                "Aggregation_Level": "Project-pooled macro average",
                "Metric": metric,
                "Projects_Total": total,
                "Projects_With_Both_Classes": both_classes,
                "Projects_Contributing": len(values),
                "Mean": values.mean() if len(values) else math.nan,
                "Median": values.median() if len(values) else math.nan,
                "Std": values.std(ddof=1) if len(values) > 1 else math.nan,
                "Min": values.min() if len(values) else math.nan,
                "Q1": values.quantile(0.25) if len(values) else math.nan,
                "Q3": values.quantile(0.75) if len(values) else math.nan,
                "Max": values.max() if len(values) else math.nan,
            })
    return pd.DataFrame(rows)


def analyze_prediction_robustness(
    predictions: pd.DataFrame, model_name: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return pooled project metrics, project-version metrics, and project summaries."""
    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")
    _validate_predictions(predictions)
    projects = _group_metrics(predictions, ["Project"], model_name)
    versions = _group_metrics(predictions, ["Project", "Version"], model_name)
    return projects, versions, summarize_project_robustness(projects)


def generate_robustness_analysis(
    run_dir: str | Path, output_dir: str | Path | None = None
) -> Path:
    """Analyze saved RF and Dummy predictions without retraining either model."""
    run_dir = Path(run_dir)
    rf_path = run_dir / "predictions.csv"
    dummy_path = run_dir / "dummy_baseline" / "predictions.csv"
    if not rf_path.is_file():
        raise FileNotFoundError(f"Random Forest predictions not found: {rf_path}")
    if not dummy_path.is_file():
        raise FileNotFoundError(f"Dummy predictions not found: {dummy_path}")
    rf = pd.read_csv(rf_path, low_memory=False)
    dummy = pd.read_csv(dummy_path, low_memory=False)
    _validate_predictions(rf)
    _validate_predictions(dummy)
    validate_prediction_alignment(rf, dummy)

    rf_project, rf_version, rf_summary = analyze_prediction_robustness(rf, "RandomForest")
    du_project, du_version, du_summary = analyze_prediction_robustness(dummy, "DummyClassifier")
    destination = Path(output_dir) if output_dir is not None else run_dir / "robustness"
    destination.mkdir(parents=True, exist_ok=True)
    pd.concat([rf_project, du_project], ignore_index=True).to_csv(
        destination / "project_robustness.csv", index=False
    )
    pd.concat([rf_version, du_version], ignore_index=True).to_csv(
        destination / "version_robustness.csv", index=False
    )
    pd.concat([rf_summary, du_summary], ignore_index=True).to_csv(
        destination / "robustness_summary.csv", index=False
    )
    notes = [
        "Diagnostics use saved out-of-project predictions; no models are retrained.",
        "Project metrics pool raw predictions across versions within each held-out project.",
        "Version metrics retain missing Version values as a separate group.",
        "robustness_summary.csv Mean is the equal-project macro average of project-pooled metrics.",
        "The workbook Project Summary instead averages version-level metrics within each project.",
        "The workbook Project Pooled sheet reports pooled project metrics and their equal-project macro average.",
        "No dataset-wide pooled metric across all held-out projects is reported.",
        "Recall is buggy-class recall; specificity is non-buggy-class recall.",
        "Balanced accuracy is undefined when the actual group has one class.",
        "MCC is undefined when the actual or predicted labels contain one class.",
        "Undefined values remain missing and are omitted from summary statistics.",
    ]
    (destination / "README.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return destination
