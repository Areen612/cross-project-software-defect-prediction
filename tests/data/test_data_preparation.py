import pandas as pd
import pytest

from pandas.testing import assert_frame_equal, assert_series_equal

from src.config import FEATURES, TARGET_COLUMN, METADATA_COLUMNS
from src.data.data_preparation import prepare_modeling_data


def make_valid_dataframe():
    data = {
        feature: [1, 2, 3]
        for feature in FEATURES
    }

    data[TARGET_COLUMN] = [False, True, False]
    data["Project"] = ["Project A", "Project A", "Project B"]
    data["Version"] = ["1.0", "1.1", "2.0"]
    data["Class"] = ["ClassA", "ClassB", "ClassC"]

    return pd.DataFrame(data)


def test_returns_correct_features():
    df = make_valid_dataframe()

    X, _, _ = prepare_modeling_data(df)

    assert X.columns.tolist() == FEATURES
    assert len(X.columns) == 18


def test_returns_correct_target_as_series():
    df = make_valid_dataframe()

    _, y, _ = prepare_modeling_data(df)

    assert isinstance(y, pd.Series)
    assert y.name == TARGET_COLUMN
    assert y.tolist() == [False, True, False]


def test_returns_correct_metadata():
    df = make_valid_dataframe()

    _, _, metadata = prepare_modeling_data(df)

    assert metadata.columns.tolist() == METADATA_COLUMNS
    assert metadata["Project"].tolist() == [
        "Project A", "Project A", "Project B"
    ]


def test_raises_error_when_required_columns_are_missing():
    df = make_valid_dataframe()

    df = df.drop(columns=["noc", "Version"])

    with pytest.raises(ValueError) as exc_info:
        prepare_modeling_data(df)

    assert "noc" in str(exc_info.value)
    assert "Version" in str(exc_info.value)


def test_does_not_modify_original_dataframe():
    df = make_valid_dataframe()
    original_df = df.copy(deep=True)

    prepare_modeling_data(df)

    assert_frame_equal(df, original_df)


def test_outputs_are_independent_of_original_dataframe():
    df = make_valid_dataframe()

    X, y, metadata = prepare_modeling_data(df)

    X.loc[0, "noc"] = 999
    y.iloc[0] = True
    metadata.loc[0, "Project"] = "Modified"

    assert df.loc[0, "noc"] == 1
    assert df.loc[0, TARGET_COLUMN] == False
    assert df.loc[0, "Project"] == "Project A"