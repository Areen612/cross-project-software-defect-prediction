from src.models.random_forest import create_random_forest
from src.preprocessing.imputation import impute_fold_data


def train_fold(
    X,
    y,
    metadata,
    held_out_project,
    train_indices,
    test_indices,
    model_factory=None,
):
    """
    Train and evaluate one LOPO fold using the supplied estimator factory.

    Random Forest remains the default model. Alternative classifiers can be
    supplied through model_factory without changing the split or preprocessing
    protocol. Imputation is always fitted on the training partition only.
    """
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    metadata_test = metadata.iloc[test_indices]

    (
        X_train_imputed,
        X_test_imputed,
        imputer,
        dropped_features,
    ) = impute_fold_data(X_train, X_test)

    # Resolve the factory at call time so tests and callers can override it.
    estimator_factory = (
        create_random_forest if model_factory is None else model_factory
    )
    model = estimator_factory()
    model.fit(X_train_imputed, y_train)
    predictions = model.predict(X_test_imputed)

    return {
        "held_out_project": held_out_project,
        "y_test": y_test,
        "predictions": predictions,
        "model": model,
        "imputer": imputer,
        "dropped_features": dropped_features,
        "metadata_test": metadata_test,
    }
