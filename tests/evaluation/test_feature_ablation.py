import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.feature_ablation import (
    ABLATION_CONFIGURATIONS,
    _build_ablation_predictions,
    _evaluate_variant,
    _validate_configurations,
    run_feature_ablation,
)


def _tiny_data():
    X = pd.DataFrame({
        "loc": [1, 3, 5, 7, 9, 11],
        "noc": [0, 1, 2, 3, 4, 5],
        "cbo": [2, 3, 2, 3, 2, 3],
    })
    y = pd.Series([False, True, False, True, False, True], name="buggy")
    metadata = pd.DataFrame({
        "Project": ["A", "A", "B", "B", "C", "C"],
        "Version": ["1", "1", "2", "2", "3", "3"],
        "Class": ["A1", "A2", "B1", "B2", "C1", "C2"],
    })
    return X, y, metadata


def test_default_ablation_configurations_are_documented_and_valid():
    _validate_configurations(ABLATION_CONFIGURATIONS)
    assert ABLATION_CONFIGURATIONS["baseline_all_features"] == []
    assert ABLATION_CONFIGURATIONS["without_loc"] == ["loc"]
    assert ABLATION_CONFIGURATIONS["without_noc"] == ["noc"]
    assert ABLATION_CONFIGURATIONS["without_loc_noc"] == ["loc", "noc"]


@pytest.mark.parametrize(
    "configurations",
    [
        {},
        {"": []},
        {"bad": "loc"},
        {"unknown": ["not_a_feature"]},
        {"duplicate": ["loc", "loc"]},
        {"all": ["noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", "lcom3",
                 "loc", "dam", "moa", "mfa", "cam", "ic", "cbm", "amc", "nr", "ndc"]},
    ],
)
def test_invalid_ablation_configurations_are_rejected(configurations):
    with pytest.raises(ValueError):
        _validate_configurations(configurations)


def test_ablation_builds_one_prediction_per_observation_and_lopo_fold():
    X, y, metadata = _tiny_data()
    predictions, folds, runtime = _build_ablation_predictions(
        X, y, metadata, ["loc"]
    )
    assert len(predictions) == len(X)
    assert len(folds) == metadata["Project"].nunique()
    assert set(predictions.columns) == {
        "Project", "Version", "Class", "Actual", "Predicted"
    }
    assert runtime >= 0


def test_ablation_uses_only_retained_predictors():
    X, y, metadata = _tiny_data()
    predictions, folds, _ = _build_ablation_predictions(
        X, y, metadata, ["loc"]
    )
    assert len(predictions) == len(X)
    for fold in folds:
        assert "loc" not in fold["model"].feature_names_in_


def test_variant_evaluation_returns_project_version_and_macro_metrics():
    X, y, metadata = _tiny_data()
    predictions, _, _ = _build_ablation_predictions(X, y, metadata, [])
    version, project, macro = _evaluate_variant(predictions)
    assert len(version) == 3
    assert len(project) == 3
    assert len(macro) == 1


def test_run_feature_ablation_writes_isolated_artifacts(tmp_path, monkeypatch):
    from src.evaluation import feature_ablation as module

    X, y, metadata = _tiny_data()
    raw = pd.DataFrame({
        "bugs": [0, 1, 0, 1, 0, 1],
        "Project": metadata["Project"],
        "Version": metadata["Version"],
        "Class": metadata["Class"],
        "loc": X["loc"],
        "noc": X["noc"],
        "cbo": X["cbo"],
    })
    # The real experiment expects all configured predictors, so build remaining
    # required columns with deterministic numeric values.
    from src.config import FEATURES
    for index, feature in enumerate(FEATURES):
        if feature not in raw:
            raw[feature] = np.arange(len(raw), dtype=float) + index
    monkeypatch.setattr(module, "load_dataset", lambda: raw.copy())
    monkeypatch.setattr(module, "validate_dataset", lambda frame: None)

    reference_dir = tmp_path / "rf_run"
    reference_dir.mkdir()
    expected = metadata.copy()
    expected["Actual"] = y
    expected["Predicted"] = y
    expected.to_csv(reference_dir / "predictions.csv", index=False)
    (reference_dir / "metadata.json").write_text("{}", encoding="utf-8")

    output = run_feature_ablation(
        reference_dir,
        configurations={"all": [], "no_loc": ["loc"]},
    )
    assert output == reference_dir / "feature_ablation"
    assert (output / "project_metrics_all.csv").is_file()
    assert (output / "version_metrics_all.csv").is_file()
    assert (output / "macro_metrics_all.csv").is_file()
    manifest = json.loads((output / "metadata.json").read_text())
    assert set(manifest["configurations"]) == {"all", "no_loc"}
    assert (output / "no_loc" / "predictions.csv").is_file()


def test_run_rejects_missing_reference_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_feature_ablation(tmp_path / "missing")


def test_run_rejects_reference_without_predictions(tmp_path):
    run_dir = tmp_path / "rf_run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        run_feature_ablation(run_dir)
