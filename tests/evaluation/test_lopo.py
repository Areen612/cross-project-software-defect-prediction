
import pandas as pd
import pytest

from src.evaluation.lopo import generate_lopo_splits


def make_metadata():
    return pd.DataFrame({
        "Project": [
            "Project A", "Project A",
            "Project B", "Project B",
            "Project C", "Project C",
        ],
        "Version": ["1.0", "1.1", "1.0", "1.1", "1.0", "1.1"],
        "Class": ["A1", "A2", "B1", "B2", "C1", "C2"],
    })


def test_generates_one_fold_per_project():
    metadata = make_metadata()

    splits = generate_lopo_splits(metadata)

    assert len(splits) == metadata["Project"].nunique()


def test_each_project_is_held_out_exactly_once():
    metadata = make_metadata()

    splits = generate_lopo_splits(metadata)

    held_out_projects = [
        project for project, _, _ in splits
    ]

    assert set(held_out_projects) == set(metadata["Project"].unique())
    assert len(held_out_projects) == len(set(held_out_projects))


def test_training_and_test_projects_are_disjoint():
    metadata = make_metadata()

    splits = generate_lopo_splits(metadata)

    for project, train_indices, test_indices in splits:
        train_projects = set(
            metadata.iloc[train_indices]["Project"]
        )
        test_projects = set(
            metadata.iloc[test_indices]["Project"]
        )

        assert test_projects == {project}
        assert project not in train_projects
        assert train_projects.isdisjoint(test_projects)


def test_every_observation_is_in_test_once():
    metadata = make_metadata()

    splits = generate_lopo_splits(metadata)

    all_test_indices = [
        index
        for _, _, test_indices in splits
        for index in test_indices
    ]

    assert sorted(all_test_indices) == list(range(len(metadata)))


def test_training_and_test_indices_cover_all_observations():
    metadata = make_metadata()

    splits = generate_lopo_splits(metadata)

    for _, train_indices, test_indices in splits:
        assert len(train_indices) + len(test_indices) == len(metadata)
        assert set(train_indices).isdisjoint(set(test_indices))


def test_raises_error_when_project_column_is_missing():
    metadata = make_metadata().drop(columns=["Project"])

    with pytest.raises(ValueError, match="Project"):
        generate_lopo_splits(metadata)


def test_raises_error_when_metadata_is_empty():
    metadata = pd.DataFrame({
        "Project": [],
    })

    with pytest.raises(ValueError, match="no observations"):
        generate_lopo_splits(metadata)


def test_raises_error_when_project_identifiers_are_missing():
    metadata = make_metadata()
    metadata.loc[0, "Project"] = None

    with pytest.raises(ValueError, match="Missing project identifiers"):
        generate_lopo_splits(metadata)


def test_raises_error_when_only_one_project_exists():
    metadata = pd.DataFrame({
        "Project": ["Project A", "Project A"],
    })

    with pytest.raises(ValueError, match="at least two"):
        generate_lopo_splits(metadata)
