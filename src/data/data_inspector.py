import pandas as pd

from src.config import FEATURES, SOURCE_COLUMN
from src.data.data_loader import load_dataset, validate_dataset


def inspect_dataset_overview(df: pd.DataFrame) -> None:
    print("\n--- Dataset overview ---")
    print(f"Shape: {df.shape}")
    print(f"Projects: {df['Project'].nunique()}")
    print(f"Versions: {df['Version'].nunique()}")


def inspect_feature_types(df: pd.DataFrame) -> None:
    print("\n--- Predictor data types ---")
    print(df[FEATURES].dtypes.to_string())


def inspect_missing_values(df: pd.DataFrame) -> None:
    print("\n--- Missing predictor values ---")
    print(df[FEATURES].isna().sum().to_string())


def inspect_target_distribution(df: pd.DataFrame) -> None:
    print("\n--- Target distribution ---")
    print(df[SOURCE_COLUMN].value_counts(dropna=False).to_string())


def inspect_project_distribution(df: pd.DataFrame) -> None:
    print("\n--- Observations per project ---")
    print(df.groupby("Project").size().describe().to_string())


def main() -> None:
    df = load_dataset()
    validate_dataset(df)

    inspect_dataset_overview(df)
    inspect_feature_types(df)
    inspect_missing_values(df)
    inspect_target_distribution(df)
    inspect_project_distribution(df)


if __name__ == "__main__":
    main()