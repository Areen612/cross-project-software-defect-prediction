
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    _calculate_pooled_mcc,
    aggregate_project_metrics,
    evaluate_predictions,
)


@pytest.fixture
def sample_predictions():
    """Predictions containing multiple projects and versions."""
    return pd.DataFrame({
        "Project": [
            "A", "A", "A", "A",
            "A", "A",
            "B", "B",
        ],
        "Version": [
            "1.0", "1.0", "1.0", "1.0",
            "1.1", "1.1",
            "2.0", "2.0",
        ],
        "Actual": [
            True, True, False, False,
            True, False,
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
    """Version-level metrics for testing project aggregation."""
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


# ---------------------------------------------------------------------
# evaluate_predictions()
# ---------------------------------------------------------------------

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

    version_a = result[
        (result["Project"] == "A")
        & (result["Version"] == "1.0")
    ].iloc[0]

    assert version_a["TN"] == 1
    assert version_a["FP"] == 1
    assert version_a["FN"] == 1
    assert version_a["TP"] == 1


def test_evaluate_predictions_calculates_metrics(sample_predictions):
    result = evaluate_predictions(sample_predictions)

    version_a = result[
        (result["Project"] == "A")
        & (result["Version"] == "1.0")
    ].iloc[0]

    assert version_a["Accuracy"] == pytest.approx(0.5)
    assert version_a["Precision"] == pytest.approx(0.5)
    assert version_a["Recall"] == pytest.approx(0.5)
    assert version_a["F1"] == pytest.approx(0.5)
    assert version_a["MCC"] == pytest.approx(0.0)


def test_evaluate_predictions_calculates_support_and_class_counts(
    sample_predictions,
):
    result = evaluate_predictions(sample_predictions)

    version_a = result[
        (result["Project"] == "A")
        & (result["Version"] == "1.0")
    ].iloc[0]

    assert version_a["Support"] == 4
    assert version_a["Buggy_Count"] == 2
    assert version_a["Non_Buggy_Count"] == 2


def test_evaluate_predictions_marks_informative_mcc(sample_predictions):
    result = evaluate_predictions(sample_predictions)

    version_a = result[
        (result["Project"] == "A")
        & (result["Version"] == "1.0")
    ].iloc[0]

    assert version_a["MCC_Status"] == "Informative"


def test_evaluate_predictions_marks_actual_single_class_as_not_informative():
    predictions = pd.DataFrame({
        "Project": ["A"] * 3,
        "Version": ["1.0"] * 3,
        "Actual": [True, True, True],
        "Predicted": [True, False, True],
    })

    result = evaluate_predictions(predictions)

    assert result.loc[0, "MCC_Status"] == "Not informative"
    assert result.loc[0, "MCC"] == pytest.approx(0.0)


def test_evaluate_predictions_marks_predicted_single_class_as_undefined():
    predictions = pd.DataFrame({
        "Project": ["A"] * 4,
        "Version": ["1.0"] * 4,
        "Actual": [True, True, False, False],
        "Predicted": [True, True, True, True],
    })

    result = evaluate_predictions(predictions)

    assert result.loc[0, "MCC_Status"] == "Undefined"
    assert result.loc[0, "MCC"] == pytest.approx(0.0)


def test_evaluate_predictions_preserves_missing_versions():
    predictions = pd.DataFrame({
        "Project": ["A", "A"],
        "Version": [None, None],
        "Actual": [True, False],
        "Predicted": [True, False],
    })

    result = evaluate_predictions(predictions)

    assert len(result) == 1
    assert pd.isna(result.loc[0, "Version"])


def test_evaluate_predictions_raises_error_for_empty_dataframe():
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        evaluate_predictions(pd.DataFrame())


def test_evaluate_predictions_raises_error_for_missing_columns():
    predictions = pd.DataFrame({
        "Project": ["A"],
        "Actual": [True],
    })

    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_predictions(predictions)


def test_evaluate_predictions_raises_error_for_missing_required_values():
    predictions = pd.DataFrame({
        "Project": ["A"],
        "Version": ["1.0"],
        "Actual": [None],
        "Predicted": [True],
    })

    with pytest.raises(ValueError, match="cannot contain missing values"):
        evaluate_predictions(predictions)


def test_evaluate_predictions_raises_error_for_missing_project():
    predictions = pd.DataFrame({
        "Project": [None],
        "Version": ["1.0"],
        "Actual": [True],
        "Predicted": [True],
    })

    with pytest.raises(ValueError, match="cannot contain missing values"):
        evaluate_predictions(predictions)


def test_evaluate_predictions_raises_error_for_non_boolean_labels():
    predictions = pd.DataFrame({
        "Project": ["A"],
        "Version": ["1.0"],
        "Actual": [1],
        "Predicted": [1],
    })

    with pytest.raises(ValueError, match="must have boolean dtype"):
        evaluate_predictions(predictions)


# ---------------------------------------------------------------------
# _calculate_pooled_mcc()
# ---------------------------------------------------------------------

def test_calculate_pooled_mcc():
    result = _calculate_pooled_mcc(
        tn=1,
        fp=1,
        fn=1,
        tp=3,
    )

    assert result == pytest.approx(0.25)


def test_calculate_pooled_mcc_returns_zero_when_denominator_is_zero():
    result = _calculate_pooled_mcc(
        tn=0,
        fp=0,
        fn=2,
        tp=3,
    )

    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------
# aggregate_project_metrics()
# ---------------------------------------------------------------------

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
    assert set(result["Project"]) == {"A", "B"}


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

    expected = (
        (0.5 * 4) + (1.0 * 2)
    ) / 6

    assert project_a["Accuracy_Weighted"] == pytest.approx(expected)
    assert project_a["Precision_Weighted"] == pytest.approx(expected)
    assert project_a["Recall_Weighted"] == pytest.approx(expected)
    assert project_a["F1_Weighted"] == pytest.approx(expected)


def test_weighted_and_unweighted_metrics_can_differ(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    project_a = result[result["Project"] == "A"].iloc[0]

    assert project_a["Accuracy"] != pytest.approx(
        project_a["Accuracy_Weighted"]
    )


def test_aggregate_project_metrics_sums_support_and_class_counts(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    project_a = result[result["Project"] == "A"].iloc[0]

    assert project_a["Support"] == 6
    assert project_a["Buggy_Count"] == 4
    assert project_a["Non_Buggy_Count"] == 2


def test_aggregate_project_metrics_counts_versions(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    project_a = result[result["Project"] == "A"].iloc[0]

    assert project_a["Version_Count"] == 2


def test_aggregate_project_metrics_excludes_non_informative_mcc(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    project_a = result[result["Project"] == "A"].iloc[0]

    # Only Version 1.0 contributes to the mean MCC.
    assert project_a["MCC"] == pytest.approx(0.0)
    assert project_a["MCC_Informative_Versions"] == 1


def test_aggregate_project_metrics_excludes_undefined_and_non_informative_mcc():
    version_metrics = pd.DataFrame({
        "Project": ["A", "A", "A"],
        "Version": ["1.0", "1.1", "1.2"],
        "Support": [4, 4, 4],
        "Buggy_Count": [2, 4, 2],
        "Non_Buggy_Count": [2, 0, 2],
        "Accuracy": [0.75, 1.0, 0.5],
        "Precision": [0.75, 1.0, 0.5],
        "Recall": [0.75, 1.0, 0.5],
        "F1": [0.75, 1.0, 0.5],
        "MCC": [0.5, 0.0, 0.0],
        "MCC_Status": [
            "Informative",
            "Not informative",
            "Undefined",
        ],
        "TN": [1, 0, 0],
        "FP": [1, 0, 2],
        "FN": [1, 0, 2],
        "TP": [1, 4, 0],
    })

    result = aggregate_project_metrics(version_metrics)

    assert result.loc[0, "MCC"] == pytest.approx(0.5)
    assert result.loc[0, "MCC_Informative_Versions"] == 1


def test_mean_mcc_is_nan_when_no_version_is_informative():
    version_metrics = pd.DataFrame({
        "Project": ["A", "A"],
        "Version": ["1.0", "1.1"],
        "Support": [4, 4],
        "Buggy_Count": [4, 2],
        "Non_Buggy_Count": [0, 2],
        "Accuracy": [1.0, 0.5],
        "Precision": [1.0, 0.5],
        "Recall": [1.0, 0.5],
        "F1": [1.0, 0.5],
        "MCC": [0.0, 0.0],
        "MCC_Status": ["Not informative", "Undefined"],
        "TN": [0, 0],
        "FP": [0, 2],
        "FN": [0, 2],
        "TP": [4, 0],
    })

    result = aggregate_project_metrics(version_metrics)

    assert pd.isna(result.loc[0, "MCC"])
    assert result.loc[0, "MCC_Informative_Versions"] == 0


def test_aggregate_project_metrics_sums_confusion_counts(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    project_a = result[result["Project"] == "A"].iloc[0]

    assert project_a["TN"] == 1
    assert project_a["FP"] == 1
    assert project_a["FN"] == 1
    assert project_a["TP"] == 3


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


def test_mean_mcc_and_pooled_mcc_are_separate_calculations():
    version_metrics = pd.DataFrame({
        "Project": ["A", "A"],
        "Version": ["1.0", "1.1"],
        "Support": [10, 100],
        "Buggy_Count": [5, 50],
        "Non_Buggy_Count": [5, 50],
        "Accuracy": [0.5, 0.9],
        "Precision": [0.5, 0.9],
        "Recall": [0.5, 0.9],
        "F1": [0.5, 0.9],
        "MCC": [0.2, 0.9],
        "MCC_Status": ["Informative", "Informative"],
        "TN": [3, 45],
        "FP": [2, 5],
        "FN": [2, 5],
        "TP": [3, 45],
    })

    result = aggregate_project_metrics(version_metrics)

    # Mean MCC gives both versions equal weight.
    assert result.loc[0, "MCC"] == pytest.approx(0.55)

    # Pooled MCC uses the combined confusion counts.
    expected_pooled = _calculate_pooled_mcc(
        tn=48,
        fp=7,
        fn=7,
        tp=48,
    )

    assert result.loc[0, "Pooled_MCC"] == pytest.approx(expected_pooled)
    assert result.loc[0, "MCC"] != pytest.approx(
        result.loc[0, "Pooled_MCC"]
    )


@pytest.mark.parametrize(
    (
        "buggy_count",
        "non_buggy_count",
        "tn",
        "fp",
        "fn",
        "tp",
        "expected",
    ),
    [
        # Actual labels contain only one class.
        (4, 0, 0, 0, 0, 4, "Not informative"),

        # Actual labels contain both classes, predictions only one class.
        (2, 2, 0, 4, 0, 0, "Undefined"),

        # Actual and predicted labels both contain two classes.
        (2, 2, 1, 1, 1, 1, "Informative"),
    ],
)
def test_pooled_mcc_status(
    buggy_count,
    non_buggy_count,
    tn,
    fp,
    fn,
    tp,
    expected,
):
    version_metrics = pd.DataFrame({
        "Project": ["A"],
        "Version": ["1.0"],
        "Support": [buggy_count + non_buggy_count],
        "Buggy_Count": [buggy_count],
        "Non_Buggy_Count": [non_buggy_count],
        "Accuracy": [0.5],
        "Precision": [0.5],
        "Recall": [0.5],
        "F1": [0.5],
        "MCC": [0.0],
        "MCC_Status": [expected],
        "TN": [tn],
        "FP": [fp],
        "FN": [fn],
        "TP": [tp],
    })

    result = aggregate_project_metrics(version_metrics)

    assert result.loc[0, "Pooled_MCC_Status"] == expected


def test_aggregate_project_metrics_does_not_modify_input(
    sample_version_metrics,
):
    original = sample_version_metrics.copy(deep=True)

    aggregate_project_metrics(sample_version_metrics)

    pd.testing.assert_frame_equal(sample_version_metrics, original)


def test_aggregate_project_metrics_does_not_return_temporary_columns(
    sample_version_metrics,
):
    result = aggregate_project_metrics(sample_version_metrics)

    temporary_columns = [
        column
        for column in result.columns
        if column.endswith("_Weighted_Numerator")
    ]

    assert temporary_columns == []


def test_aggregate_project_metrics_preserves_missing_version():
    version_metrics = pd.DataFrame({
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

    result = aggregate_project_metrics(version_metrics)

    assert len(result) == 1
    assert result.loc[0, "Version_Count"] == 1


def test_aggregate_project_metrics_raises_error_for_empty_dataframe():
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        aggregate_project_metrics(pd.DataFrame())


def test_aggregate_project_metrics_raises_error_for_missing_columns():
    version_metrics = pd.DataFrame({
        "Project": ["A"],
        "Version": ["1.0"],
    })

    with pytest.raises(ValueError, match="missing required columns"):
        aggregate_project_metrics(version_metrics)


def test_aggregate_project_metrics_raises_error_for_missing_project(
    sample_version_metrics,
):
    sample_version_metrics.loc[0, "Project"] = None

    with pytest.raises(ValueError, match="Project cannot contain missing"):
        aggregate_project_metrics(sample_version_metrics)


def test_aggregate_project_metrics_raises_error_for_duplicate_project_version(
    sample_version_metrics,
):
    duplicate = pd.concat(
        [sample_version_metrics, sample_version_metrics.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate Project-Version"):
        aggregate_project_metrics(duplicate)


def test_aggregate_project_metrics_raises_error_for_missing_numeric_values(
    sample_version_metrics,
):
    sample_version_metrics.loc[0, "Accuracy"] = np.nan

    with pytest.raises(ValueError, match="cannot contain missing values"):
        aggregate_project_metrics(sample_version_metrics)


def test_aggregate_project_metrics_raises_error_for_non_numeric_metrics(
    sample_version_metrics,
):
    sample_version_metrics["Accuracy"] = (
        sample_version_metrics["Accuracy"].astype(str)
    )

    with pytest.raises(ValueError, match="must be numeric"):
        aggregate_project_metrics(sample_version_metrics)


def test_aggregate_project_metrics_raises_error_for_zero_support(
    sample_version_metrics,
):
    sample_version_metrics.loc[0, "Support"] = 0

    with pytest.raises(ValueError, match="Support must be greater than zero"):
        aggregate_project_metrics(sample_version_metrics)


def test_aggregate_project_metrics_rejects_invalid_mcc_status(
    sample_version_metrics,
):
    sample_version_metrics.loc[0, "MCC_Status"] = "Invalid"

    with pytest.raises(ValueError, match="MCC_Status must contain only"):
        aggregate_project_metrics(sample_version_metrics)