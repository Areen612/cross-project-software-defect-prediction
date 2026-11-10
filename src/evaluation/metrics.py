
import math

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

VERSION_METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
]

VALID_MCC_STATUSES = {
    "Informative",
    "Not informative",
    "Undefined",
}


def _get_mcc_status(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> str:
    """
    Determine whether MCC is informative or mathematically undefined.

    Returns:
        Not informative:
            Actual labels contain only one class.

        Undefined:
            Actual labels contain both classes, but predicted
            labels contain only one class.

        Informative:
            Both actual and predicted labels contain both classes.
    """
    if y_true.nunique() < 2:
        return "Not informative"

    if y_pred.nunique() < 2:
        return "Undefined"

    return "Informative"


def _calculate_pooled_mcc(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
) -> float:
    """
    Calculate MCC from combined confusion matrix counts.

    Returns zero when the denominator is zero, following
    scikit-learn's MCC convention.
    """
    numerator = (tp * tn) - (fp * fn)

    denominator = math.sqrt(
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _get_pooled_mcc_status(row: pd.Series) -> str:
    """
    Determine the MCC status from combined project-level counts.

    Actual class counts are checked first, followed by predicted
    class counts.
    """
    if (
        row["Buggy_Count"] == 0
        or row["Non_Buggy_Count"] == 0
    ):
        return "Not informative"

    if (
        row["TP"] + row["FP"] == 0
        or row["TN"] + row["FN"] == 0
    ):
        return "Undefined"

    return "Informative"


def evaluate_predictions(
    predictions_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate classification metrics for each Project-Version group.

    Missing versions are retained as separate groups.

    Parameters:
        predictions_dataframe:
            DataFrame containing Project, Version, Actual, and Predicted.

    Returns:
        DataFrame containing one row per Project-Version group,
        with classification metrics and confusion matrix counts.
    """
    if predictions_dataframe.empty:
        raise ValueError(
            "Input DataFrame is empty. Cannot evaluate predictions."
        )

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
        if not pd.api.types.is_bool_dtype(
            predictions_dataframe[column]
        ):
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

        mcc_status = _get_mcc_status(y_true, y_pred)

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

    Primary metrics:
        Accuracy, Precision, Recall, and F1 are averaged equally
        across versions.
        MCC is averaged across informative versions only.

    Supplementary metrics:
        Accuracy, Precision, Recall, and F1 are also calculated
        as support-weighted means across versions.
        Pooled MCC is calculated from the combined confusion counts.

    Counts are summed across versions.

    Parameters:
        version_metrics_dataframe:
            DataFrame containing one row per Project-Version group
            and its evaluation metrics.

    Returns:
        DataFrame containing one row per project, with unweighted
        metrics, weighted metrics, pooled MCC, statuses, and counts.
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

    if (version_metrics_dataframe["Support"] <= 0).any():
        raise ValueError("Support must be greater than zero.")

    if not version_metrics_dataframe["MCC_Status"].isin(
        VALID_MCC_STATUSES
    ).all():
        raise ValueError(
            "MCC_Status must contain only 'Informative', "
            "'Not informative', or 'Undefined'."
        )

    working_dataframe = version_metrics_dataframe.copy()

    # Exclude non-informative and undefined MCC values from the mean.
    working_dataframe["MCC"] = working_dataframe["MCC"].where(
        working_dataframe["MCC_Status"] == "Informative"
    )

    # Count only versions that contribute to the mean MCC.
    working_dataframe["MCC_Informative_Flag"] = (
        working_dataframe["MCC_Status"] == "Informative"
    )

    # Prepare numerator columns for support-weighted metrics.
    for metric in VERSION_METRICS:
        working_dataframe[f"{metric}_Weighted_Numerator"] = (
            working_dataframe[metric] * working_dataframe["Support"]
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

            # Primary: unweighted mean across versions.
            Accuracy=("Accuracy", "mean"),
            Precision=("Precision", "mean"),
            Recall=("Recall", "mean"),
            F1=("F1", "mean"),
            MCC=("MCC", "mean"),

            MCC_Informative_Versions=("MCC_Informative_Flag", "sum"),

            # Supplementary: weighted metric numerators.
            Accuracy_Weighted_Numerator=(
                "Accuracy_Weighted_Numerator", "sum"
            ),
            Precision_Weighted_Numerator=(
                "Precision_Weighted_Numerator", "sum"
            ),
            Recall_Weighted_Numerator=(
                "Recall_Weighted_Numerator", "sum"
            ),
            F1_Weighted_Numerator=(
                "F1_Weighted_Numerator", "sum"
            ),

            # Combined confusion matrix counts.
            TN=("TN", "sum"),
            FP=("FP", "sum"),
            FN=("FN", "sum"),
            TP=("TP", "sum"),
        )
    )

    # Calculate support-weighted means.
    for metric in VERSION_METRICS:
        numerator_column = f"{metric}_Weighted_Numerator"
        weighted_column = f"{metric}_Weighted"

        project_metrics_dataframe[weighted_column] = (
            project_metrics_dataframe[numerator_column]
            / project_metrics_dataframe["Support"]
        )

        project_metrics_dataframe = project_metrics_dataframe.drop(
            columns=numerator_column
        )

    # Calculate pooled MCC from the combined confusion counts.
    project_metrics_dataframe["Pooled_MCC"] = (
        project_metrics_dataframe.apply(
            lambda row: _calculate_pooled_mcc(
                tn=int(row["TN"]),
                fp=int(row["FP"]),
                fn=int(row["FN"]),
                tp=int(row["TP"]),
            ),
            axis=1,
        )
    )

    # Determine pooled MCC status from combined actual and predicted counts.
    project_metrics_dataframe["Pooled_MCC_Status"] = (
        project_metrics_dataframe.apply(
            _get_pooled_mcc_status,
            axis=1,
        )
    )

    project_metrics_dataframe["MCC_Informative_Versions"] = (
        project_metrics_dataframe["MCC_Informative_Versions"].astype(int)
    )

    return project_metrics_dataframe


def calculate_macro_average(
    project_metrics_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate macro-average metrics across projects.

    Every project contributes equally, regardless of its number
    of versions or class observations.

    MCC:
        Averages project-level MCC values only for projects
        with at least one informative version.

    Pooled MCC:
        Averages project-level pooled MCC values only for projects
        whose pooled MCC status is informative.

    Returns:
        A one-row DataFrame containing macro-average metrics
        and the number of projects contributing to each MCC result.
    """
    if project_metrics_dataframe.empty:
        raise ValueError(
            "Input DataFrame is empty. Cannot calculate macro average."
        )

    required_columns = {
        "Project",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Accuracy_Weighted",
        "Precision_Weighted",
        "Recall_Weighted",
        "F1_Weighted",
        "MCC",
        "MCC_Informative_Versions",
        "Pooled_MCC",
        "Pooled_MCC_Status",
    }

    missing_columns = required_columns - set(
        project_metrics_dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Input DataFrame is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if project_metrics_dataframe["Project"].isnull().any():
        raise ValueError("Project cannot contain missing values.")

    if project_metrics_dataframe["Project"].duplicated().any():
        raise ValueError(
            "Input DataFrame must contain one row per project."
        )

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Accuracy_Weighted",
        "Precision_Weighted",
        "Recall_Weighted",
        "F1_Weighted",
    ]

    non_numeric_columns = [
        column
        for column in metric_columns
        if not pd.api.types.is_numeric_dtype(
            project_metrics_dataframe[column]
        )
    ]

    if non_numeric_columns:
        raise ValueError(
            f"Metric columns must be numeric: {non_numeric_columns}"
        )

    valid_pooled_statuses = VALID_MCC_STATUSES

    if not project_metrics_dataframe["Pooled_MCC_Status"].isin(
        valid_pooled_statuses
    ).all():
        raise ValueError(
            "Pooled_MCC_Status must contain only 'Informative', "
            "'Not informative', or 'Undefined'."
        )

    # Every project contributes equally to these metrics.
    macro_results = {
        "Project_Count": len(project_metrics_dataframe),
    }

    for metric in metric_columns:
        macro_results[metric] = (
            project_metrics_dataframe[metric].mean()
        )

    # Mean project-level MCC, excluding projects without
    # an informative project-level MCC.
    informative_mcc_projects = project_metrics_dataframe[
        project_metrics_dataframe["MCC_Informative_Versions"] > 0
    ]

    macro_results["MCC"] = informative_mcc_projects["MCC"].mean()

    macro_results["MCC_Informative_Projects"] = len(
        informative_mcc_projects
    )

    # Mean of project-level pooled MCC values, including only
    # projects with informative pooled MCC.
    informative_pooled_projects = project_metrics_dataframe[
        project_metrics_dataframe["Pooled_MCC_Status"] == "Informative"
    ]

    macro_results["Pooled_MCC"] = (
        informative_pooled_projects["Pooled_MCC"].mean()
    )

    macro_results["Pooled_MCC_Informative_Projects"] = len(
        informative_pooled_projects
    )

    return pd.DataFrame([macro_results])