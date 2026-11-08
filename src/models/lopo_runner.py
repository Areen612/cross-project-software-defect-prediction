from src.evaluation.lopo import generate_lopo_splits
from src.models.fold_trainer import train_fold


def run_lopo_experiment(X, y, metadata):
    """
    Run a Leave-One-Project-Out (LOPO) experiment.

    Parameters
    ----------
    X : pandas.DataFrame
        Predictor features.
    y : pandas.Series
        Binary target labels.
    metadata : pandas.DataFrame
        Observation metadata, including Project, Version, and Class.

    Returns
    -------
    list[dict]
        A list of dictionaries containing results for each fold.
    """
    # Generate one split per project
    splits = generate_lopo_splits(metadata)

    # Store the results of all folds
    results = []

    # Train and predict for each held-out project
    for held_out_project, train_indices, test_indices in splits:
        fold_results = train_fold(
            X,
            y,
            metadata,
            held_out_project,
            train_indices,
            test_indices,
        )

        results.append(fold_results)

    return results