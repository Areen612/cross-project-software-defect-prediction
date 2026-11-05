import pandas as pd

from src.config import SOURCE_COLUMN, TARGET_COLUMN
from src.data.target_validation import validate_bug_counts


def target_builder(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a binary target variable from the original bug-count column.

    Args:
        df: Input DataFrame containing the bug-count column.

    Returns:
        A copy of the DataFrame with the binary target column added.

    Raises:
        ValueError: If the bug-count column is missing, contains invalid
                    values, or contains negative values.
    """
    if SOURCE_COLUMN not in df.columns:
        raise ValueError(
            f"Required target source column '{SOURCE_COLUMN}' is missing."
        )

    if TARGET_COLUMN in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' already exists. "
            "Remove it before building the target."
        )
    
    # Validate and convert bug counts using shared validation
    bug_counts = validate_bug_counts(df[SOURCE_COLUMN])

    # Create a copy to preserve the original DataFrame
    result = df.copy()

    # A class is buggy if its bug count is greater than zero
    result[TARGET_COLUMN] = bug_counts > 0

    return result