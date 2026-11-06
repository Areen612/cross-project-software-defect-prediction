
import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from src.preprocessing.imputation import impute_fold_data


def test_imputes_missing_values_using_training_medians():
    X_train = pd.DataFrame({
        "loc": [10, np.nan, 30],
        "cbo": [2, 4, 6],
    })

    X_test = pd.DataFrame({
        "loc": [np.nan, 100],
        "cbo": [8, np.nan],
    })

    X_train_imputed, X_test_imputed, _, _ = impute_fold_data(
        X_train, X_test
    )

    expected_train = pd.DataFrame({
        "loc": [10.0, 20.0, 30.0],
        "cbo": [2.0, 4.0, 6.0],
    })

    expected_test = pd.DataFrame({
        "loc": [20.0, 100.0],
        "cbo": [8.0, 4.0],
    })

    assert_frame_equal(X_train_imputed, expected_train)
    assert_frame_equal(X_test_imputed, expected_test)


def test_imputer_learns_statistics_from_training_data_only():
    X_train = pd.DataFrame({
        "loc": [10, 20, 30],
    })

    X_test = pd.DataFrame({
        "loc": [100, 200, np.nan],
    })

    _, X_test_imputed, imputer, _ = impute_fold_data(
        X_train, X_test
    )

    # Training median is 20, not the test median.
    assert imputer.statistics_[0] == 20
    assert X_test_imputed.loc[2, "loc"] == 20


def test_drops_features_entirely_missing_in_training():
    X_train = pd.DataFrame({
        "loc": [10, 20, 30],
        "avg_cc": [np.nan, np.nan, np.nan],
    })

    X_test = pd.DataFrame({
        "loc": [40, np.nan],
        "avg_cc": [100, 200],
    })

    X_train_imputed, X_test_imputed, _, dropped_features = (
        impute_fold_data(X_train, X_test)
    )

    assert dropped_features == ["avg_cc"]
    assert X_train_imputed.columns.tolist() == ["loc"]
    assert X_test_imputed.columns.tolist() == ["loc"]


def test_preserves_dataframe_indices_and_feature_names():
    X_train = pd.DataFrame(
        {"loc": [10, np.nan, 30]},
        index=[5, 10, 15],
    )

    X_test = pd.DataFrame(
        {"loc": [np.nan, 50]},
        index=[20, 25],
    )

    X_train_imputed, X_test_imputed, _, _ = impute_fold_data(
        X_train, X_test
    )

    assert X_train_imputed.index.tolist() == [5, 10, 15]
    assert X_test_imputed.index.tolist() == [20, 25]

    assert X_train_imputed.columns.tolist() == ["loc"]
    assert X_test_imputed.columns.tolist() == ["loc"]


def test_does_not_modify_original_dataframes():
    X_train = pd.DataFrame({
        "loc": [10, np.nan, 30],
        "cbo": [2, 4, 6],
    })

    X_test = pd.DataFrame({
        "loc": [np.nan, 50],
        "cbo": [8, 10],
    })

    original_train = X_train.copy(deep=True)
    original_test = X_test.copy(deep=True)

    impute_fold_data(X_train, X_test)

    assert_frame_equal(X_train, original_train)
    assert_frame_equal(X_test, original_test)


def test_raises_error_when_feature_columns_differ():
    X_train = pd.DataFrame({
        "loc": [10, 20],
        "cbo": [2, 4],
    })

    X_test = pd.DataFrame({
        "loc": [30, 40],
        "rfc": [5, 6],
    })

    with pytest.raises(ValueError, match="identical columns"):
        impute_fold_data(X_train, X_test)


def test_raises_error_when_feature_order_differs():
    X_train = pd.DataFrame({
        "loc": [10, 20],
        "cbo": [2, 4],
    })

    X_test = pd.DataFrame({
        "cbo": [5, 6],
        "loc": [30, 40],
    })

    with pytest.raises(ValueError, match="identical columns"):
        impute_fold_data(X_train, X_test)
