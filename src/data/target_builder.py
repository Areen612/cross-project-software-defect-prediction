import numpy as np
import pandas as pd

from src.config import SOURCE_COLUMN, TARGET_COLUMN


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

    # Convert bug counts to numeric values for validation.
    bug_counts = pd.to_numeric(df[SOURCE_COLUMN], errors="coerce")

    # Reject missing, non-numeric, and non-finite values.
    if not np.isfinite(bug_counts).all():
        raise ValueError(
            "Missing or invalid values found in the bug-count column."
        )

    # Reject negative bug counts.
    if (bug_counts < 0).any():
        raise ValueError(
            "Negative values found in the bug-count column."
        )

    # Create a copy to avoid modifying the original DataFrame.
    result = df.copy()

    # Construct the binary target.
    result[TARGET_COLUMN] = bug_counts > 0

    return result