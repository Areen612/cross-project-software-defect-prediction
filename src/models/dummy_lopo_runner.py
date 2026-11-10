"""Run the DummyClassifier baseline over the project's standard LOPO splits."""

from src.evaluation.lopo import generate_lopo_splits
from src.models.dummy_baseline import train_dummy_fold


def run_dummy_lopo_experiment(X, y, metadata):
    """Run one most-frequent DummyClassifier fold per held-out project."""
    splits = generate_lopo_splits(metadata)
    results = []

    for held_out_project, train_indices, test_indices in splits:
        results.append(
            train_dummy_fold(
                X,
                y,
                metadata,
                held_out_project,
                train_indices,
                test_indices,
            )
        )

    return results
