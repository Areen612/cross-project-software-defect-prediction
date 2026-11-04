import pandas as pd
from pandas import DataFrame

from src.config import INPUT_PATH


def load_dataset() -> DataFrame:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at: {INPUT_PATH}"
        )

    if INPUT_PATH.stat().st_size == 0:
        raise ValueError(
            f"Dataset file is empty: {INPUT_PATH}"
        )

    return pd.read_csv(INPUT_PATH)