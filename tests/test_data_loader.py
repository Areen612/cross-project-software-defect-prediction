
import pandas as pd
import pytest

from src import config
from src.data import data_loader


def make_valid_dataframe() -> pd.DataFrame:
    """
    Build a small synthetic DataFrame containing all required columns.

    We use synthetic data so tests do not depend on the full CSV.
    """
    required_columns = (
        [config.SOURCE_COLUMN]
        + config.METADATA_COLUMNS
        + config.FEATURES
    )

    data = {
        column: [1, 2]
        for column in required_columns
    }

    # Give metadata and target columns realistic example values.
    data["Project"] = ["Tomcat", "JEdit"]
    data["Version"] = ["1.0", "2.0"]
    data["Class"] = ["ClassA", "ClassB"]
    data[config.SOURCE_COLUMN] = [1, 0]

    return pd.DataFrame(data)


# --------------------------------------------------
# Tests for load_dataset()
# --------------------------------------------------

def test_load_dataset_returns_dataframe(tmp_path, monkeypatch):
    """The loader should read a valid CSV and return a DataFrame."""

    csv_path = tmp_path / "test_dataset.csv"

    expected_df = make_valid_dataframe()
    expected_df.to_csv(csv_path, index=False)

    # Replace the configured input path with our temporary CSV.
    monkeypatch.setattr(data_loader, "INPUT_PATH", csv_path)

    result = data_loader.load_dataset()

    assert isinstance(result, pd.DataFrame)
    assert result.shape == expected_df.shape
    assert list(result.columns) == list(expected_df.columns)


def test_load_dataset_raises_error_when_file_is_missing(
    tmp_path, monkeypatch
):
    """A missing dataset should raise FileNotFoundError."""

    csv_path = tmp_path / "missing.csv"

    monkeypatch.setattr(data_loader, "INPUT_PATH", csv_path)

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        data_loader.load_dataset()


def test_load_dataset_raises_error_when_file_is_empty(
    tmp_path, monkeypatch
):
    """A zero-byte dataset should raise ValueError."""

    csv_path = tmp_path / "empty.csv"
    csv_path.touch()

    monkeypatch.setattr(data_loader, "INPUT_PATH", csv_path)

    with pytest.raises(ValueError, match="Dataset file is empty"):
        data_loader.load_dataset()


# --------------------------------------------------
# Tests for validate_dataset()
# --------------------------------------------------

def test_validate_dataset_accepts_valid_dataframe():
    """A DataFrame with all required columns and valid target passes."""

    df = make_valid_dataframe()

    result = data_loader.validate_dataset(df)

    assert result is None


def test_validate_dataset_rejects_empty_dataframe():
    """An empty DataFrame should raise ValueError."""

    df = pd.DataFrame(
        columns=(
            [config.SOURCE_COLUMN]
            + config.METADATA_COLUMNS
            + config.FEATURES
        )
    )

    with pytest.raises(ValueError, match="no observations"):
        data_loader.validate_dataset(df)


def test_validate_dataset_rejects_duplicate_columns():
    """Duplicate column names should raise ValueError."""

    df = make_valid_dataframe()

    # Add a second column with the same name.
    df = df.assign(Project_copy=df["Project"])
    df = df.rename(columns={"Project_copy": "Project"})

    with pytest.raises(ValueError, match="Duplicate column names"):
        data_loader.validate_dataset(df)


def test_validate_dataset_rejects_missing_required_column():
    """A missing required predictor should raise ValueError."""

    df = make_valid_dataframe()
    df = df.drop(columns=["noc"])

    with pytest.raises(
        ValueError,
        match="Required columns missing"
    ):
        data_loader.validate_dataset(df)


@pytest.mark.parametrize(
    "invalid_value",
    [
        None,       # Missing target
        "unknown",  # Non-numeric target
        -1,         # Negative target
    ],
)
def test_validate_dataset_rejects_invalid_target_values(invalid_value):
    """Missing, non-numeric, and negative target values are rejected."""

    df = make_valid_dataframe()

    # Convert the target column to object dtype so it can hold
    # both numeric and non-numeric test values.
    df[config.SOURCE_COLUMN] = df[config.SOURCE_COLUMN].astype(object)

    df.loc[0, config.SOURCE_COLUMN] = invalid_value

    with pytest.raises(ValueError):
        data_loader.validate_dataset(df)


def test_validate_dataset_accepts_numeric_string_targets():
    """Numeric strings should be accepted as valid target values."""

    df = make_valid_dataframe()
    df[config.SOURCE_COLUMN] = ["1", "0"]

    result = data_loader.validate_dataset(df)

    assert result is None


def test_validate_dataset_warns_about_unexpected_columns():
    """Unexpected columns should generate a warning, not an error."""

    df = make_valid_dataframe()
    df["unexpected_column"] = [10, 20]

    with pytest.warns(
        UserWarning,
        match="Unexpected columns found"
    ):
        data_loader.validate_dataset(df)


def test_load_dataset_treats_dash_as_missing(tmp_path, monkeypatch):
    csv_path = tmp_path / "test_dataset.csv"

    csv_path.write_text(
        "Project,Version,Class,bugs,noc\n"
        "Tomcat,1.0,ClassA,1,10\n"
        "JEdit,2.0,ClassB,0,-\n"
    )

    monkeypatch.setattr(data_loader, "INPUT_PATH", csv_path)

    result = data_loader.load_dataset()

    assert pd.isna(result.loc[1, "noc"])