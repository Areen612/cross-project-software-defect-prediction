
from src.config import (
    EXCLUDED_FEATURES,
    FEATURES,
    METADATA_COLUMNS,
    BUG_COUNT_COLUMN,
)


def test_primary_features_match_approved_feature_list():
    """The primary model must use exactly the 18 approved predictors."""

    expected_features = [
        "noc","cbo","rfc","lcom","ca","ce",
        "npm","lcom3","loc","dam","moa","mfa",
        "cam","ic","cbm","amc","nr","ndc",
    ]

    assert FEATURES == expected_features
    assert len(FEATURES) == 18


def test_excluded_features_match_supervisor_instruction():
    """All six supervisor-excluded variables must remain excluded."""

    expected_excluded = [
        "nml",
        "isOld",
        "ndpv",
        "ner",
        "max(cc)",
        "avg(cc)",
    ]

    assert EXCLUDED_FEATURES == expected_excluded


def test_excluded_features_are_not_primary_predictors():
    """No supervisor-excluded variable may enter the primary model."""

    assert set(FEATURES).isdisjoint(EXCLUDED_FEATURES)


def test_metadata_and_target_are_not_predictors():
    """Identifiers and the target source must not enter the feature list."""

    forbidden_columns = set(METADATA_COLUMNS + [BUG_COUNT_COLUMN])

    assert set(FEATURES).isdisjoint(forbidden_columns)
