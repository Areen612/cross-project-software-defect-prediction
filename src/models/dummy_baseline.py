"""LOPO training pipeline for the most-frequent DummyClassifier baseline."""

from sklearn.dummy import DummyClassifier

from src.preprocessing.imputation import impute_fold_data


def train_dummy_fold(X, y, metadata, held_out_project, train_indices, test_indices):
    """Train a most-frequent DummyClassifier using one LOPO training fold.

    The same fold-local median imputation and training/test split as the
    Random Forest pipeline are retained for an apples-to-apples workflow.
    The DummyClassifier itself does not use the predictor values.
    """
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    metadata_test = metadata.iloc[test_indices]

    _, _, imputer, dropped_features = impute_fold_data(X_train, X_test)

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train.iloc[:, :0], y_train)
    predictions = model.predict(X_test.iloc[:, :0])

    return {
        "held_out_project": held_out_project,
        "y_test": y_test,
        "predictions": predictions,
        "model": model,
        "imputer": imputer,
        "dropped_features": dropped_features,
        "metadata_test": metadata_test,
    }
