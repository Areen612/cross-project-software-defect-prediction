
import pandas as pd
import pytest

from src.config import SOURCE_COLUMN, TARGET_COLUMN
from src.data.target_builder import target_builder


def test_target_builder_creates_binary_target():
    """Test that target_builder creates a binary target column correctly."""

    df = pd.DataFrame({
        SOURCE_COLUMN: [0, 1, 2, 0, 3]
    })

    result = target_builder(df)

    expected_target = [False, True, True, False, True]

    assert list(result[TARGET_COLUMN]) == expected_target
    assert pd.api.types.is_bool_dtype(result[TARGET_COLUMN])


def test_preserve_original_dataframe():
    """Test that target_builder does not modify the original DataFrame."""

    df = pd.DataFrame({
        SOURCE_COLUMN: [0, 1, 2]
    })

    original_df = df.copy()

    result = target_builder(df)

    pd.testing.assert_frame_equal(df, original_df)
    pd.testing.assert_series_equal(
        result[SOURCE_COLUMN],
        original_df[SOURCE_COLUMN]
    )
        
    assert TARGET_COLUMN not in df.columns
    assert TARGET_COLUMN in result.columns
    assert SOURCE_COLUMN in result.columns



def test_target_builder_rejects_missing_values():
    """Test that target_builder rejects missing bug counts."""

    df = pd.DataFrame({
        SOURCE_COLUMN: [0, 1, None, 3]
    })

    with pytest.raises(ValueError, match="Missing or invalid values"):
        target_builder(df)


def test_target_builder_rejects_non_numeric_values():
    """Test that target_builder rejects non-numeric bug counts."""

    df = pd.DataFrame({
        SOURCE_COLUMN: [0, 1, "unknown", 3]
    })

    with pytest.raises(ValueError, match="Missing or invalid values"):
        target_builder(df)


def test_target_builder_rejects_negative_values():
    """Test that target_builder rejects negative bug counts."""

    df = pd.DataFrame({
        SOURCE_COLUMN: [0, 1, -2, 3]
    })

    with pytest.raises(ValueError, match="Negative values"):
        target_builder(df)


def test_target_builder_rejects_missing_source_column():
    """Test that target_builder rejects a DataFrame without the bugs column."""

    df = pd.DataFrame({
        "Project": ["Tomcat", "JEdit"],
        "Version": ["1.0", "2.0"]
    })

    with pytest.raises(ValueError, match="Required target source column"):
        target_builder(df)
