"""Validation helpers for comparing predictions from the same observations."""

from __future__ import annotations

from collections import Counter

import pandas as pd


_ALIGNMENT_COLUMNS = ["Project", "Version", "Class", "Actual"]


def _actual_label(value: object) -> bool:
    """Normalize a binary actual label or reject ambiguous values."""
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"Actual labels must be boolean or binary 0/1; got {value!r}.")


def _observation_signatures(predictions: pd.DataFrame) -> Counter:
    missing = set(_ALIGNMENT_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Prediction data is missing alignment columns: {sorted(missing)}"
        )

    signatures = []
    for project, version, class_name, actual in predictions[
        _ALIGNMENT_COLUMNS
    ].itertuples(index=False, name=None):
        signature = (
            "<MISSING>" if pd.isna(project) else str(project),
            "<MISSING>" if pd.isna(version) else str(version),
            "<MISSING>" if pd.isna(class_name) else str(class_name),
            _actual_label(actual),
        )
        signatures.append(signature)

    return Counter(signatures)


def validate_prediction_alignment(
    reference_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
) -> None:
    """
    Require identical Project-Version-Class observations and actual labels.

    Row order is ignored, while duplicate rows are counted. Predicted labels
    are deliberately excluded because they are the values being compared.
    """
    reference = _observation_signatures(reference_predictions)
    candidate = _observation_signatures(candidate_predictions)

    if reference != candidate:
        missing_from_candidate = reference - candidate
        extra_in_candidate = candidate - reference
        raise ValueError(
            "Prediction datasets are not aligned: the Project-Version-Class "
            "observations or their Actual labels differ "
            f"({sum(missing_from_candidate.values())} missing, "
            f"{sum(extra_in_candidate.values())} unexpected)."
        )
