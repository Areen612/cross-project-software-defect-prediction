
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)


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

    # 1. Group observations by Project and Version.
    grouped = predictions_dataframe.groupby(
        ["Project", "Version"],
        dropna=False,
        sort=False,
    )

    # 2. Store the evaluation results for every group.
    evaluation_results = []

    # 3. Process each Project-Version group independently.
    for group_name, group_data in grouped:

        # Extract the project and version identifiers.
        project, version = group_name

        # Extract actual labels and model predictions.
        y_true = group_data["Actual"]
        y_pred = group_data["Predicted"]

        # 4. Calculate confusion matrix counts.
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[False, True],
        ).ravel()

        # 5. Calculate classification metrics.
        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0, 
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        mcc = matthews_corrcoef(y_true, y_pred)

        # 6. Check whether MCC is informative for two-class
        # discrimination based on the actual labels.
        if y_true.nunique() < 2:
            mcc_status = "Not informative"
        else:
            mcc_status = "Informative"

        # 7. Store all results for this group.
        evaluation_results.append({
            "Project": project,
            "Version": version,
            "Support": len(group_data),
            "Buggy_Count": int(y_true.sum()),
            "Non_Buggy_Count": int((~y_true).sum()),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MCC": mcc,
            "MCC_Status": mcc_status,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        })

    # 8. Convert the list of dictionaries into a DataFrame.
    metrics_dataframe = pd.DataFrame(evaluation_results)

    return metrics_dataframe
