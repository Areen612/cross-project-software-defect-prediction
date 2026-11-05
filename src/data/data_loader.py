import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame

from src.config import FEATURES, INPUT_PATH, METADATA_COLUMNS, SOURCE_COLUMN


def load_dataset() -> DataFrame:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at: {INPUT_PATH}"
        )

    if INPUT_PATH.stat().st_size == 0:
        raise ValueError(
            f"Dataset file is empty: {INPUT_PATH}"
        )
        
    # low_memory=False is used to avoid dtype inference issues with large datasets
    return pd.read_csv(
        INPUT_PATH,
        low_memory=False,
        na_values=["-"],
    )

def validate_dataset(df: DataFrame) -> None:
    # 1. Check duplicate column names
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {duplicates}")

    # 2. Check that the dataset contains observations
    if df.empty:
        raise ValueError("Dataset contains no observations")

    # 3. Check required columns
    required_columns = [SOURCE_COLUMN] + METADATA_COLUMNS + FEATURES

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")

    # 4. Validate target values
    target = pd.to_numeric(df[SOURCE_COLUMN], errors="coerce")

    if target.isnull().any():
        raise ValueError("Missing or non-numeric values found in target column")

    if (target < 0).any():
        raise ValueError("Negative values found in target column")

    # 5. Report unexpected columns
    unexpected_columns = [
        col for col in df.columns
        if col not in required_columns
    ]

    if unexpected_columns:
        warnings.warn(
            f"Unexpected columns found: {unexpected_columns}",
            UserWarning
        )
    
    # 6. Validate predictor features
    for col in FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' is not numeric")

        if np.isinf(df[col]).any():
            raise ValueError(f"Infinite values found in column '{col}'")