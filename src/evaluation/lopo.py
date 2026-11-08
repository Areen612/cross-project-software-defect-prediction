
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut


def generate_lopo_splits(
    metadata: pd.DataFrame,
) -> list[tuple[object, np.ndarray, np.ndarray]]:
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

    groups = metadata["Project"].to_numpy()

    logo = LeaveOneGroupOut()

    splits = []

    for train_indices, test_indices in logo.split(
        metadata,
        groups=groups,
    ):
        held_out_project = groups[test_indices[0]]

        splits.append(
            (held_out_project, train_indices, test_indices)
        )

    return splits
