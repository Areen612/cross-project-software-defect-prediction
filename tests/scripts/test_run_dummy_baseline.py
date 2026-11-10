import pandas as pd
import pytest

from scripts.run_dummy_baseline import _validate_rf_reference


def _reference_data():
    X = pd.DataFrame({"loc": [10, 20, 30]})
    y = pd.Series([True, False, True], name="buggy")
    metadata = pd.DataFrame({
        "Project": ["A", "A", "B"],
        "Version": ["1", None, "2"],
        "Class": ["A1", "A2", "B1"],
    })
    saved_rf = metadata.copy()
    saved_rf["Actual"] = y
    saved_rf["Predicted"] = [True, True, False]
    return X, y, metadata, saved_rf


def test_reference_validation_accepts_same_rows_in_different_order(tmp_path):
    X, y, metadata, saved_rf = _reference_data()
    run_dir = tmp_path / "rf_run"
    run_dir.mkdir()
    saved_rf.iloc[::-1].to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "metadata.json").write_text("{}", encoding="utf-8")

    _validate_rf_reference(run_dir, X, y, metadata)


def test_reference_validation_rejects_changed_actual_label(tmp_path):
    X, y, metadata, saved_rf = _reference_data()
    run_dir = tmp_path / "rf_run"
    run_dir.mkdir()
    saved_rf.loc[0, "Actual"] = False
    saved_rf.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Actual labels differ"):
        _validate_rf_reference(run_dir, X, y, metadata)


def test_reference_validation_rejects_changed_observation_count(tmp_path):
    X, y, metadata, saved_rf = _reference_data()
    run_dir = tmp_path / "rf_run"
    run_dir.mkdir()
    saved_rf.iloc[:-1].to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="observation count"):
        _validate_rf_reference(run_dir, X, y, metadata)
