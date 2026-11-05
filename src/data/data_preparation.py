import pandas as pd

from src.config import FEATURES, TARGET_COLUMN, METADATA_COLUMNS


def validate_data_columns(df: pd.DataFrame) -> None:
    required_columns = FEATURES + [TARGET_COLUMN] + METADATA_COLUMNS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def prepare_modeling_data(df: pd.DataFrame):
    validate_data_columns(df)

    X = df.loc[:, FEATURES].copy()
    y = df.loc[:, TARGET_COLUMN].copy()
    metadata = df.loc[:, METADATA_COLUMNS].copy()

    return X, y, metadata