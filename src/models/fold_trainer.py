
from src.models.random_forest import create_random_forest
from src.preprocessing.imputation import impute_fold_data


def train_fold(X, y, metadata, held_out_project, train_indices, test_indices):
    """
    Train and evaluate predictions for one Leave-One-Project-Out (LOPO) fold.

    The function extracts training and test observations using positional
    indices, fits a median imputer on training features only, transforms
    both datasets, trains a Random Forest, and predicts the held-out project.

    Parameters
    ----------
    X : pandas.DataFrame
        Predictor features.
    y : pandas.Series
        Binary target labels.
    metadata : pandas.DataFrame
        Observation metadata, including Project, Version, and Class.
    held_out_project : str
        Name of the project excluded from training.
    train_indices : array-like
        Positional indices of training observations.
    test_indices : array-like
        Positional indices of test observations.

    Returns
    -------
    dict
        Contains the held-out project, actual labels, predictions,
        trained model, fitted imputer, dropped features, and test metadata.
    """

    # 1. Extract training and test data
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    metadata_test = metadata.iloc[test_indices]

    # 2. Impute missing values
    (
        X_train_imputed,
        X_test_imputed,
        imputer,
        dropped_features,
    ) = impute_fold_data(X_train, X_test)

    # 3. Create the Random Forest model
    model = create_random_forest()

    # 4. Train the model using training data only
    model.fit(X_train_imputed, y_train)

    # 5. Generate predictions for the held-out project
    predictions = model.predict(X_test_imputed)

    # 6. Return the fold results
    return {
        "held_out_project": held_out_project,
        "y_test": y_test,
        "predictions": predictions,
        "model": model,
        "imputer": imputer,
        "dropped_features": dropped_features,
        "metadata_test": metadata_test,
    }
