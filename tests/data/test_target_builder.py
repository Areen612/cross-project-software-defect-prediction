
import pandas as pd
import pytest

from src.config import BUG_COUNT_COLUMN, TARGET_COLUMN
from src.data.target_builder import target_builder


def test_target_builder_creates_binary_target():
    """Test that target_builder creates a binary target column correctly."""

    df = pd.DataFrame({
        BUG_COUNT_COLUMN: [0, 1, 2, 0, 3]
    })

    result = target_builder(df)

    expected_target = [False, True, True, False, True]

    assert list(result[TARGET_COLUMN]) == expected_target
    assert pd.api.types.is_bool_dtype(result[TARGET_COLUMN])


def test_preserve_original_dataframe():
    """Test that target_builder does not modify the original DataFrame."""

    df = pd.DataFrame({
        BUG_COUNT_COLUMN: [0, 1, 2]
    })

    original_df = df.copy()

    result = target_builder(df)

    pd.testing.assert_frame_equal(df, original_df)
    pd.testing.assert_series_equal(
        result[BUG_COUNT_COLUMN],
        original_df[BUG_COUNT_COLUMN]
    )
        
    assert TARGET_COLUMN not in df.columns
    assert TARGET_COLUMN in result.columns
    assert BUG_COUNT_COLUMN in result.columns



def test_target_builder_rejects_missing_values():
    """Test that target_builder rejects missing bug counts."""

    df = pd.DataFrame({
        BUG_COUNT_COLUMN: [0, 1, None, 3]
    })

    with pytest.raises(ValueError, match="Missing or invalid values"):
        target_builder(df)


def test_target_builder_rejects_non_numeric_values():
    """Test that target_builder rejects non-numeric bug counts."""

    df = pd.DataFrame({
        BUG_COUNT_COLUMN: [0, 1, "unknown", 3]
    })

    with pytest.raises(ValueError, match="Missing or invalid values"):
        target_builder(df)


def test_target_builder_rejects_negative_values():
    """Test that target_builder rejects negative bug counts."""

    df = pd.DataFrame({
        BUG_COUNT_COLUMN: [0, 1, -2, 3]
    })

    with pytest.raises(ValueError, match="Negative values"):
        target_builder(df)


def test_target_builder_rejects_missing_BUG_COUNT_COLUMN():
    """Test that target_builder rejects a DataFrame without the bugs column."""

    df = pd.DataFrame({
        "Project": ["Tomcat", "JEdit"],
        "Version": ["1.0", "2.0"]
    })

    with pytest.raises(ValueError, match="Required target source column"):
        target_builder(df)


def test_target_builder_rejects_existing_target_column():
    """Test that target_builder rejects an existing target column."""

    df = pd.DataFrame({
        BUG_COUNT_COLUMN: [0, 1, 3],
        TARGET_COLUMN: [True, False, True]
    })

    with pytest.raises(ValueError, match="already exists"):
        target_builder(df)