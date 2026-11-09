
import pandas as pd
import pytest

from src.evaluation.metrics import evaluate_predictions


@pytest.fixture
def sample_predictions():
    return pd.DataFrame({
        "Project": ["A", "A", "A", "A", "A", "B", "B"],
        "Version": ["1.0", "1.0", "1.0", "1.0", "1.1", "2.0", "2.0"],
        "Class": [
            "C1", "C2", "C3", "C4", "C5", "C6", "C7"
        ],
        "Actual": [
            True, True, False, False, True, True, True
        ],
        "Predicted": [
            True, False, True, False, True, True, False
        ],
    })


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


def test_evaluate_predictions_calculates_metrics(
    sample_predictions,
):
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


def test_evaluate_predictions_preserves_missing_versions():
    predictions = pd.DataFrame({
        "Project": ["A", "A"],
        "Version": [None, None],
        "Class": ["C1", "C2"],
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
        "Class": ["C1", "C2"],
        "Actual": [True, True],
        "Predicted": [True, False],
    })

    result = evaluate_predictions(predictions)

    assert result.iloc[0]["Recall"] == pytest.approx(0.5)
    assert result.iloc[0]["Precision"] == pytest.approx(1.0)
    assert result.iloc[0]["MCC"] == pytest.approx(0.0)
