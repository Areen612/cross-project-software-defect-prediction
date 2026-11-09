
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)


VERSION_INPUT_COLUMNS = {
    "Project",
    "Version",
    "Actual",
    "Predicted",
}

AGGREGATION_INPUT_COLUMNS = {
    "Project",
    "Version",
    "Support",
    "Buggy_Count",
    "Non_Buggy_Count",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "MCC",
    "MCC_Status",
    "TN",
    "FP",
    "FN",
    "TP",
}

NUMERIC_METRIC_COLUMNS = [
    "Support",
    "Buggy_Count",
    "Non_Buggy_Count",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "MCC",
    "TN",
    "FP",
    "FN",
    "TP",
]


def evaluate_predictions(
    predictions_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate classification metrics for each Project-Version group.

    Missing versions are retained as separate groups.

    Parameters:
        predictions_dataframe: DataFrame containing Project, Version,
            Actual, and Predicted columns.

    Returns:
        DataFrame containing one row per Project-Version group,
        with classification metrics and confusion matrix counts.
    """
    if predictions_dataframe.empty:
        raise ValueError("Input DataFrame is empty. Cannot evaluate predictions.")

    missing_columns = VERSION_INPUT_COLUMNS - set(
        predictions_dataframe.columns
    )
    if missing_columns:
        raise ValueError(
            f"Input DataFrame is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    required_values = predictions_dataframe[
        ["Project", "Actual", "Predicted"]
    ]

    if required_values.isnull().any().any():
        raise ValueError(
            "Project, Actual, and Predicted cannot contain missing values."
        )

    for column in ["Actual", "Predicted"]:
        if not pd.api.types.is_bool_dtype(predictions_dataframe[column]):
            raise ValueError(
                f"Column '{column}' must have boolean dtype."
            )

    grouped = predictions_dataframe.groupby(
        ["Project", "Version"],
        dropna=False,
        sort=False,
    )

    evaluation_results = []

    for (project, version), group_data in grouped:
        y_true = group_data["Actual"]
        y_pred = group_data["Predicted"]

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[False, True],
        ).ravel()

        if y_true.nunique() < 2:
            mcc_status = "Not informative"
        else:
            mcc_status = "Informative"

        evaluation_results.append({
            "Project": project,
            "Version": version,
            "Support": len(group_data),
            "Buggy_Count": int(y_true.sum()),
            "Non_Buggy_Count": int((~y_true).sum()),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "F1": f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "MCC_Status": mcc_status,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        })

    return pd.DataFrame(evaluation_results)


def aggregate_project_metrics(
    version_metrics_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate version-level metrics into one summary row per project.

    Classification metrics are averaged equally across versions.
    Counts are summed across versions.

    MCC is averaged only across versions marked as informative.

    Parameters:
        version_metrics_dataframe: DataFrame containing one row per
            Project-Version group and its evaluation metrics.

    Returns:
        DataFrame containing one row per project, with mean metrics,
        total counts, version count, and informative MCC version count.
    """
    if version_metrics_dataframe.empty:
        raise ValueError(
            "Input DataFrame is empty. Cannot aggregate metrics."
        )

    missing_columns = AGGREGATION_INPUT_COLUMNS - set(
        version_metrics_dataframe.columns
    )
    if missing_columns:
        raise ValueError(
            f"Input DataFrame is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if version_metrics_dataframe["Project"].isnull().any():
        raise ValueError("Project cannot contain missing values.")

    if version_metrics_dataframe.duplicated(
        subset=["Project", "Version"]
    ).any():
        raise ValueError(
            "Input DataFrame contains duplicate Project-Version pairs."
        )

    if version_metrics_dataframe[NUMERIC_METRIC_COLUMNS].isnull().any().any():
        raise ValueError(
            "Numeric metric columns cannot contain missing values."
        )

    non_numeric_columns = [
        column
        for column in NUMERIC_METRIC_COLUMNS
        if not pd.api.types.is_numeric_dtype(
            version_metrics_dataframe[column]
        )
    ]
    if non_numeric_columns:
        raise ValueError(
            f"Metric columns must be numeric: {non_numeric_columns}"
        )

    valid_mcc_statuses = {"Informative", "Not informative"}
    if not version_metrics_dataframe["MCC_Status"].isin(
        valid_mcc_statuses
    ).all():
        raise ValueError(
            "MCC_Status must contain only 'Informative' "
            "or 'Not informative'."
        )

    working_dataframe = version_metrics_dataframe.copy()

    working_dataframe["MCC_for_mean"] = working_dataframe["MCC"].where(
        working_dataframe["MCC_Status"] == "Informative"
    )

    working_dataframe["MCC_Informative_Flag"] = (
        working_dataframe["MCC_Status"] == "Informative"
    )

    project_metrics_dataframe = (
        working_dataframe
        .groupby(
            "Project",
            as_index=False,
            sort=False,
        )
        .agg(
            Version_Count=("Version", "size"),
            Support=("Support", "sum"),
            Buggy_Count=("Buggy_Count", "sum"),
            Non_Buggy_Count=("Non_Buggy_Count", "sum"),
            Accuracy=("Accuracy", "mean"),
            Precision=("Precision", "mean"),
            Recall=("Recall", "mean"),
            F1=("F1", "mean"),
            MCC=("MCC_for_mean", "mean"),
            MCC_Informative_Versions=("MCC_Informative_Flag", "sum"),
            TN=("TN", "sum"),
            FP=("FP", "sum"),
            FN=("FN", "sum"),
            TP=("TP", "sum"),
        )
    )

    project_metrics_dataframe["MCC_Informative_Versions"] = (
        project_metrics_dataframe["MCC_Informative_Versions"].astype(int)
    )

    return project_metrics_dataframe