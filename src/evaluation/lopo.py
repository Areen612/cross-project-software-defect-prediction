
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut


def generate_lopo_splits(
    metadata: pd.DataFrame,
) -> list[tuple[object, np.ndarray, np.ndarray]]:
    """
    Generate Leave-One-Project-Out (LOPO) train/test splits.

    Each fold holds out all observations belonging to one project
    for testing, while observations from all remaining projects
    are used for training.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Observation metadata containing a 'Project' column.
        Each row represents one observation and its project identifier.

    Raises
    ------
    ValueError
        If the 'Project' column is missing, metadata is empty,
        project identifiers contain missing values, or fewer than
        two distinct projects are available.

    Returns
    -------
    list[tuple[object, np.ndarray, np.ndarray]]
        A list containing one tuple per fold. Each tuple contains:
        - The held-out project identifier.
        - Positional indices of training observations.
        - Positional indices of test observations.
    """

    # Validate the metadata before generating splits.
    if "Project" not in metadata.columns:
        raise ValueError(
            "Required metadata column 'Project' is missing."
        )

    if metadata.empty:
        raise ValueError("Metadata contains no observations.")

    if metadata["Project"].isna().any():
        raise ValueError(
            "Missing project identifiers found in metadata."
        )

    if metadata["Project"].nunique() < 2:
        raise ValueError(
            "LOPO requires at least two distinct projects."
        )

    # Extract one project identifier per observation, preserving row order.
    groups = metadata["Project"].to_numpy()

    # Configure LOPO so each distinct project is held out in a separate fold.
    logo = LeaveOneGroupOut()

    # Collect each fold's held-out project and positional train/test indices.
    splits = []

    # Generate positional indices for every leave-one-project-out fold.
    for train_indices, test_indices in logo.split(
        metadata,
        groups=groups,
    ):
        # All test observations belong to the same project, so the first
        # test index is sufficient to identify the held-out project.
        held_out_project = groups[test_indices[0]]

        # Include the project identifier so downstream functions know
        # which project this fold evaluates.
        splits.append(
            (held_out_project, train_indices, test_indices)
        )

    return splits