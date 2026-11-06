
import pandas as pd
from sklearn.impute import SimpleImputer


def impute_fold_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    # Ensure training and test features have identical columns and order.
    if not X_train.columns.equals(X_test.columns):
        raise ValueError(
            "Training and test data must have identical columns "
            "in the same order."
        )

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=False,
    )

    # Preserve DataFrame output, including feature names.
    imputer.set_output(transform="pandas")

    # Learn medians from training data and impute training data.
    X_train_imputed = imputer.fit_transform(X_train)

    # Reuse training medians to impute test data.
    X_test_imputed = imputer.transform(X_test)

    # Identify features that were entirely missing in training.
    retained_features = imputer.get_feature_names_out().tolist()

    dropped_features = [
        feature
        for feature in X_train.columns
        if feature not in retained_features
    ]

    return (
        X_train_imputed,
        X_test_imputed,
        imputer,
        dropped_features,
    )
