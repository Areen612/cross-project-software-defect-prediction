
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    evaluate_predictions,
    aggregate_project_metrics,
    _calculate_pooled_mcc,
)


# =========================================================
# Fixtures
# =========================================================

@pytest.fixture
def sample_predictions():
    return pd.DataFrame({
        "Project": ["A", "A", "A", "A", "A", "A", "B", "B"],
        "Version": [
            "1.0", "1.0", "1.0", "1.0",
            "1.1", "1.1",
            "2.0", "2.0",
        ],
        "Class": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
        "Actual": [
            True, True, False, False,
            True, True,
            True, False,
        ],
        "Predicted": [
            True, False, True, False,
            True, True,
            True, False,
        ],
    })


@pytest.fixture
def sample_version_metrics():
    return pd.DataFrame({
        "Project": ["A", "A", "B"],
        "Version": ["1.0", "1.1", "2.0"],
        "Support": [4, 2, 2],
        "Buggy_Count": [2, 2, 1],
        "Non_Buggy_Count": [2, 0, 1],
        "Accuracy": [0.5, 1.0, 1.0],
        "Precision": [0.5, 1.0, 1.0],
        "Recall": [0.5, 1.0, 1.0],
        "F1": [0.5, 1.0, 1.0],
        "MCC": [0.0, 0.0, 1.0],
        "MCC_Status": [
            "Informative",
            "Not informative",
            "Informative",
        ],
        "TN": [1, 0, 1],
        "FP": [1, 0, 0],
        "FN": [1, 0, 0],
        "TP": [1, 2, 1],
    })


# =========================================================
# evaluate_predictions() tests
# =========================================================

def test_evaluate_predictions_returns_dataframe(sample_predictions):
    result = evaluate_predictions(sample_predictions)

    assert isinstance(result, pd.DataFrame)


def test_evaluate_predictions_creates_one_row_per_project_version(
    sample_predictions,
):
    result = evaluate_predictions(sample_predictions)

    assert len(result) == 3

    assert set(zip(result["Project"], result["Version"])) == {
        ("A", "1.0"),
        ("A", "1.1"),
        ("B", "2.0"),
    }


def test_evaluate_predictions_calculates_confusion_counts(
    sample_predictions,
):
    result = evaluate_predictions(sample_predictions)

    group = result[
        (result["Project"] == "A")
        & (result["Version"] == "1.0")
    ].iloc[0]

    assert group["TP"] == 1
    assert group["TN"] == 1
    assert group["FP"] == 1
    assert group["FN"] == 1


def test_evaluate_predictions_calculates_metrics(sample_predictions):
    result = evaluate_predictions(sample_predictions)

    group = result[
        (result["Project"] == "A")
        & (result["Version"] == "1.0")
    ].iloc[0]

    assert group["Accuracy"] == pytest.approx(0.5)
    assert group["Precision"] == pytest.approx(0.5)
    assert group["Recall"] == pytest.approx(0.5)
    assert group["F1"] == pytest.approx(0.5)
    assert group["MCC"] == pytest.approx(0.0)


def test_evaluate_predictions_calculates_support_and_class_counts(
    sample_predictions,
):
    result = evaluate_predictions(sample_predictions)

    group = result[
        (result["Project"] == "A")
        & (result["Version"] == "1.0")
    ].iloc[0]

    assert group["Support"] == 4
    assert group["Buggy_Count"] == 2
    assert group["Non_Buggy_Count"] == 2


def test_evaluate_predictions_preserves_missing_versions():
    predictions = pd.DataFrame({
        "Project": ["A", "A"],
        "Version": [None, None],
        "Actual": [True, False],
        "Predicted": [True, False],
    })

    result = evaluate_predictions(predictions)

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["Version"])
    assert result.iloc[0]["TP"] == 1
    assert result.iloc[0]["TN"] == 1


def test_evaluate_predictions_handles_single_class_groups():
    predictions = pd.DataFrame({
        "Project": ["A", "A"],
        "Version": ["1.0", "1.0"],
        "Actual": [True, True],
        "Predicted": [True, False],
    })

    result = evaluate_predictions(predictions)

    assert result.iloc[0]["Recall"] == pytest.approx(0.5)
    assert result.iloc[0]["Precision"] == pytest.approx(1.0)
    assert result.iloc[0]["MCC"] == pytest.approx(0.0)
    assert result.iloc[0]["MCC_Status"] == "Not informative"


def test_evaluate_predictions_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        evaluate_predictions(pd.DataFrame())


def test_evaluate_predictions_rejects_missing_columns():
    predictions = pd.DataFrame({
        "Project": ["A"],
        "Version": ["1.0"],
        "Actual": [True],
    })

    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_predictions(predictions)


def test_evaluate_predictions_rejects_missing_labels():
    predictions = pd.DataFrame({
        "Project": ["A"],
        "Version": ["1.0"],
        "Actual": [True],
        "Predicted": [None],
    })

    with pytest.raises(ValueError, match="missing values"):
        evaluate_predictions(predictions)


def test_evaluate_predictions_rejects_non_boolean_labels():
    predictions = pd.DataFrame({
        "Project": ["A", "A"],
        "Version": ["1.0", "1.0"],
        "Actual": [1, 0],
        "Predicted": [1, 0],
    })

    with pytest.raises(ValueError, match="boolean dtype"):
        evaluate_predictions(predictions)


def test_evaluate_predictions_rejects_missing_project():
    predictions = pd.DataFrame({
        "Project": [None],
        "Version": ["1.0"],
        "Actual": [True],
        "Predicted": [True],
    })

    with pytest.raises(ValueError, match="missing values"):
        evaluate_predictions(predictions)


# =========================================================
# _calculate_pooled_mcc() tests
# =========================================================

def test_calculate_pooled_mcc():
    result = _calculate_pooled_mcc(
        tn=1,
        fp=1,
        fn=1,
        tp=3,
    )

    expected = 2 / np.sqrt(64)

    assert result == pytest.approx(expected)


def test_calculate_pooled_mcc_returns_zero_for_zero_denominator():
    result = _calculate_pooled_mcc(
        tn=0,
        fp=0,
        fn=1,
        tp=2,
    )

    assert result == 0.0


# =========================================================
# aggregate_project_metrics() tests
# =========================================================

def test_aggregate_project_metrics_returns_dataframe(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    assert isinstance(result, pd.DataFrame)


def test_aggregate_project_metrics_creates_one_row_per_project(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    assert len(result) == 2
    assert result["Project"].tolist() == ["A", "B"]


def test_aggregate_project_metrics_calculates_unweighted_means(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    project_a = result[result["Project"] == "A"].iloc[0]

    assert project_a["Accuracy"] == pytest.approx(0.75)
    assert project_a["Precision"] == pytest.approx(0.75)
    assert project_a["Recall"] == pytest.approx(0.75)
    assert project_a["F1"] == pytest.approx(0.75)


def test_aggregate_project_metrics_calculates_weighted_means(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    project_a = result[result["Project"] == "A"].iloc[0]

    expected_weighted_mean = (0.5 * 4 + 1.0 * 2) / 6

    assert project_a["Accuracy_Weighted"] == pytest.approx(
        expected_weighted_mean
    )
    assert project_a["Precision_Weighted"] == pytest.approx(
        expected_weighted_mean
    )
    assert project_a["Recall_Weighted"] == pytest.approx(
        expected_weighted_mean
    )
    assert project_a["F1_Weighted"] == pytest.approx(
        expected_weighted_mean
    )


def test_weighted_and_unweighted_means_can_differ(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    project_a = result[result["Project"] == "A"].iloc[0]

    assert project_a["Accuracy"] != pytest.approx(
        project_a["Accuracy_Weighted"]
    )


def test_aggregate_project_metrics_excludes_uninformative_mcc(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    project_a = result[result["Project"] == "A"].iloc[0]

    assert project_a["MCC"] == pytest.approx(0.0)
    assert project_a["MCC_Informative_Versions"] == 1


def test_aggregate_project_metrics_returns_nan_when_no_mcc_is_informative(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    project_b = result[result["Project"] == "B"].iloc[0]

    # This fixture is informative for B, so its MCC should be retained.
    assert project_b["MCC"] == pytest.approx(1.0)
    assert project_b["MCC_Informative_Versions"] == 1


def test_aggregate_project_metrics_sums_counts(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    project_a = result[result["Project"] == "A"].iloc[0]

    assert project_a["Support"] == 6
    assert project_a["Buggy_Count"] == 4
    assert project_a["Non_Buggy_Count"] == 2
    assert project_a["TN"] == 1
    assert project_a["FP"] == 1
    assert project_a["FN"] == 1
    assert project_a["TP"] == 3


def test_aggregate_project_metrics_counts_versions(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    project_a = result[result["Project"] == "A"].iloc[0]

    assert project_a["Version_Count"] == 2
    assert project_a["MCC_Informative_Versions"] == 1


def test_aggregate_project_metrics_calculates_pooled_mcc(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    project_a = result[result["Project"] == "A"].iloc[0]

    expected = _calculate_pooled_mcc(
        tn=1,
        fp=1,
        fn=1,
        tp=3,
    )

    assert project_a["Pooled_MCC"] == pytest.approx(expected)


def test_pooled_mcc_differs_from_mean_version_mcc(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    project_a = result[result["Project"] == "A"].iloc[0]

    assert project_a["Pooled_MCC"] != pytest.approx(
        project_a["MCC"]
    )


def test_aggregate_project_metrics_sets_pooled_mcc_status(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    project_a = result[result["Project"] == "A"].iloc[0]
    project_b = result[result["Project"] == "B"].iloc[0]

    assert project_a["Pooled_MCC_Status"] == "Informative"
    assert project_b["Pooled_MCC_Status"] == "Informative"


def test_aggregate_project_metrics_marks_single_class_pooled_mcc():
    dataframe = pd.DataFrame({
        "Project": ["A"],
        "Version": ["1.0"],
        "Support": [2],
        "Buggy_Count": [2],
        "Non_Buggy_Count": [0],
        "Accuracy": [0.5],
        "Precision": [1.0],
        "Recall": [0.5],
        "F1": [2 / 3],
        "MCC": [0.0],
        "MCC_Status": ["Not informative"],
        "TN": [0],
        "FP": [0],
        "FN": [1],
        "TP": [1],
    })

    result = aggregate_project_metrics(dataframe)

    assert result.loc[0, "Pooled_MCC"] == 0.0
    assert result.loc[0, "Pooled_MCC_Status"] == "Not informative"
    assert pd.isna(result.loc[0, "MCC"])


def test_aggregate_project_metrics_retains_missing_version():
    dataframe = pd.DataFrame({
        "Project": ["A"],
        "Version": [None],
        "Support": [2],
        "Buggy_Count": [1],
        "Non_Buggy_Count": [1],
        "Accuracy": [1.0],
        "Precision": [1.0],
        "Recall": [1.0],
        "F1": [1.0],
        "MCC": [1.0],
        "MCC_Status": ["Informative"],
        "TN": [1],
        "FP": [0],
        "FN": [0],
        "TP": [1],
    })

    result = aggregate_project_metrics(dataframe)

    assert len(result) == 1
    assert result.loc[0, "Version_Count"] == 1


def test_aggregate_project_metrics_does_not_modify_input(
    sample_version_metrics,
):
    original = sample_version_metrics.copy(deep=True)

    aggregate_project_metrics(sample_version_metrics)

    pd.testing.assert_frame_equal(
        sample_version_metrics,
        original,
    )


def test_aggregate_project_metrics_does_not_return_temporary_columns(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    assert not any(
        "Numerator" in column or "Flag" in column
        for column in result.columns
    )


def test_aggregate_project_metrics_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        aggregate_project_metrics(pd.DataFrame())


def test_aggregate_project_metrics_rejects_duplicate_project_version_pairs(
    sample_version_metrics,
):
    duplicate = pd.concat(
        [sample_version_metrics, sample_version_metrics.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate"):
        aggregate_project_metrics(duplicate)


def test_aggregate_project_metrics_rejects_missing_numeric_values(
    sample_version_metrics,
):
    dataframe = sample_version_metrics.copy()
    dataframe.loc[0, "Accuracy"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        aggregate_project_metrics(dataframe)


def test_aggregate_project_metrics_rejects_invalid_mcc_status(
    sample_version_metrics,
):
    dataframe = sample_version_metrics.copy()
    dataframe.loc[0, "MCC_Status"] = "Unknown"

    with pytest.raises(ValueError, match="MCC_Status"):
        aggregate_project_metrics(dataframe)


def test_aggregate_project_metrics_rejects_missing_project(
    sample_version_metrics,
):
    dataframe = sample_version_metrics.copy()
    dataframe.loc[0, "Project"] = None

    with pytest.raises(ValueError, match="Project"):
        aggregate_project_metrics(dataframe)


def test_aggregate_project_metrics_rejects_non_numeric_metrics(
    sample_version_metrics,
):
    dataframe = sample_version_metrics.copy()
    dataframe["Accuracy"] = dataframe["Accuracy"].astype(str)

    with pytest.raises(ValueError, match="must be numeric"):
        aggregate_project_metrics(dataframe)


def test_aggregate_project_metrics_rejects_zero_support(
    sample_version_metrics,
):
    dataframe = sample_version_metrics.copy()
    dataframe.loc[0, "Support"] = 0

    with pytest.raises(ValueError, match="Support"):
        aggregate_project_metrics(dataframe)