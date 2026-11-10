from src.evaluation.lopo import generate_lopo_splits
from src.models.fold_trainer import train_fold


def run_lopo_experiment(X, y, metadata, model_factory=None):
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
    model_factory : callable, optional
        Zero-argument estimator factory. Defaults to the project's
        Random Forest factory through train_fold.
    """
    splits = generate_lopo_splits(metadata)
    results = []

    for held_out_project, train_indices, test_indices in splits:
        fold_arguments = (
            X,
            y,
            metadata,
            held_out_project,
            train_indices,
            test_indices,
        )

        # Preserve the original six-argument call when using the default
        # RF factory, including compatibility with existing monkeypatches.
        if model_factory is None:
            fold_results = train_fold(*fold_arguments)
        else:
            fold_results = train_fold(
                *fold_arguments,
                model_factory=model_factory,
            )

        results.append(fold_results)

    return results
