
import numpy as np
import pandas as pd


def validate_bug_counts(series: pd.Series) -> pd.Series:
    """
    Validate bug-count values and return them as numeric values.

    Missing values, non-numeric values, infinite values,
    and negative bug counts are rejected.
    """

    # Convert values to numeric, turning invalid strings into NaN
    numeric_series = pd.to_numeric(series, errors="coerce")

    # Reject missing and non-numeric values
    if numeric_series.isnull().any():
        raise ValueError(
            "Missing or invalid values found in bug-count column."
        )

    # Reject positive and negative infinity
    if np.isinf(numeric_series).any():
        raise ValueError(
            "Infinite values found in bug-count column."
        )

    # Reject negative bug counts
    if (numeric_series < 0).any():
        raise ValueError(
            "Negative values found in bug-count column."
        )

    return numeric_series
