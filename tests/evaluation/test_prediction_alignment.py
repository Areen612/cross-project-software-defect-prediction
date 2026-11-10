import pandas as pd
import pytest

from src.evaluation.prediction_alignment import validate_prediction_alignment


def _predictions():
    return pd.DataFrame({
        "Project": ["A", "A", "B"],
        "Version": ["1.0", None, "2.0"],
        "Class": ["A1", "A2", "B1"],
        "Actual": [True, False, True],
        "Predicted": [True, True, False],
    })


def test_alignment_accepts_same_observations_in_different_order():
    reference = _predictions()
    candidate = reference.iloc[[2, 0, 1]].reset_index(drop=True)

    validate_prediction_alignment(reference, candidate)


def test_alignment_ignores_predictions_but_checks_actual_labels():
    reference = _predictions()
    candidate = reference.copy()
    candidate["Predicted"] = [False, False, False]

    validate_prediction_alignment(reference, candidate)


def test_alignment_rejects_different_actual_labels():
    reference = _predictions()
    candidate = reference.copy()
    candidate.loc[0, "Actual"] = False

    with pytest.raises(ValueError, match="Actual labels differ"):
        validate_prediction_alignment(reference, candidate)


def test_alignment_rejects_different_class_identity():
    reference = _predictions()
    candidate = reference.copy()
    candidate.loc[0, "Class"] = "OtherClass"

    with pytest.raises(ValueError, match="observations or their Actual labels differ"):
        validate_prediction_alignment(reference, candidate)


def test_alignment_rejects_different_project_version_group():
    reference = _predictions()
    candidate = reference.copy()
    candidate.loc[0, "Version"] = "1.9"

    with pytest.raises(ValueError, match="observations or their Actual labels differ"):
        validate_prediction_alignment(reference, candidate)


def test_alignment_counts_duplicate_observations():
    reference = _predictions()
    candidate = pd.concat(
        [reference.iloc[:2], reference.iloc[:2], reference.iloc[2:]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="observations or their Actual labels differ"):
        validate_prediction_alignment(reference, candidate)


def test_alignment_requires_class_and_actual_columns():
    reference = _predictions()
    candidate = reference.drop(columns=["Class"])

    with pytest.raises(ValueError, match="alignment columns"):
        validate_prediction_alignment(reference, candidate)


def test_alignment_rejects_ambiguous_actual_labels():
    reference = _predictions()
    candidate = reference.copy()
    candidate["Actual"] = ["yes", False, True]

    with pytest.raises(ValueError, match="Actual labels must be boolean"):
        validate_prediction_alignment(reference, candidate)
