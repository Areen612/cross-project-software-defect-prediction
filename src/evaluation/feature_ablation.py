"""Feature-ablation experiments using the unchanged project-level LOPO protocol."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import sklearn

from src.config import (
    FEATURES, RANDOM_SEED, N_ESTIMATORS, MAX_FEATURES, CLASS_WEIGHT, N_JOBS
)
from src.data.data_loader import load_dataset, validate_dataset
from src.data.target_builder import target_builder
from src.data.data_preparation import prepare_modeling_data
from src.evaluation.lopo import generate_lopo_splits
from src.evaluation.metrics import evaluate_predictions, aggregate_project_metrics, calculate_macro_average
from src.models.fold_trainer import train_fold

ABLATION_CONFIGURATIONS = {
    "baseline_all_features": [],
    "without_loc": ["loc"],
    "without_noc": ["noc"],
    "without_loc_noc": ["loc", "noc"],
    "without_top_association_features": ["loc", "noc", "ca", "ce"],
}


def _validate_configurations(configurations: dict[str, list[str]]) -> None:
    if not configurations:
        raise ValueError("At least one ablation configuration is required.")
    for name, removed in configurations.items():
        if not name or not isinstance(name, str):
            raise ValueError("Ablation configuration names must be non-empty strings.")
        if not isinstance(removed, list):
            raise ValueError(f"Removed features for '{name}' must be a list.")
        unknown = sorted(set(removed) - set(FEATURES))
        if unknown:
            raise ValueError(f"Unknown features in '{name}': {unknown}")
        if len(set(removed)) != len(removed):
            raise ValueError(f"Duplicate removed features in '{name}'.")
        if len(removed) == len(FEATURES):
            raise ValueError(f"Configuration '{name}' cannot remove every feature.")


def _build_ablation_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    removed_features: list[str],
) -> tuple[pd.DataFrame, list[dict], float]:
    selected_features = [feature for feature in X.columns if feature not in removed_features]
    if not selected_features:
        raise ValueError("Ablation configuration leaves no predictors.")
    X_variant = X.loc[:, selected_features].copy()
    started = time.perf_counter()
    results = []
    for held_out_project, train_indices, test_indices in generate_lopo_splits(metadata):
        results.append(train_fold(
            X_variant, y, metadata, held_out_project, train_indices, test_indices
        ))
    runtime_seconds = round(time.perf_counter() - started, 2)
    prediction_frames = []
    for fold in results:
        meta = fold["metadata_test"].copy()
        actual = fold["y_test"].rename("Actual")
        predicted = pd.Series(fold["predictions"], index=actual.index, name="Predicted")
        prediction_frames.append(pd.concat([meta, actual, predicted], axis=1))
    predictions = pd.concat(prediction_frames).reset_index(drop=True)
    predictions = predictions[["Project", "Version", "Class", "Actual", "Predicted"]]
    if len(predictions) != len(X) or predictions[["Actual", "Predicted"]].isna().any().any():
        raise RuntimeError("Ablation prediction coverage or label validation failed.")
    from src.evaluation.prediction_alignment import validate_prediction_alignment
    expected = metadata[["Project", "Version", "Class"]].copy()
    expected["Actual"] = y.to_numpy()
    validate_prediction_alignment(expected, predictions)
    return predictions, results, runtime_seconds


def _evaluate_variant(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    version = evaluate_predictions(predictions)
    project = aggregate_project_metrics(version)
    macro = calculate_macro_average(project)
    return version, project, macro


def run_feature_ablation(
    output_dir: str | Path,
    configurations: dict[str, list[str]] | None = None,
) -> Path:
    """Run each feature set with independent LOPO training and save isolated artifacts."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Reference RF run directory does not exist: {output_dir}")
    reference_predictions_path = output_dir / "predictions.csv"
    reference_metadata_path = output_dir / "metadata.json"
    if not reference_predictions_path.is_file() or not reference_metadata_path.is_file():
        raise FileNotFoundError("Reference run must contain predictions.csv and metadata.json.")

    configurations = configurations or ABLATION_CONFIGURATIONS
    _validate_configurations(configurations)

    dataset = load_dataset()
    validate_dataset(dataset)
    prepared = target_builder(dataset)
    X, y, metadata = prepare_modeling_data(prepared)

    reference = pd.read_csv(reference_predictions_path, low_memory=False)
    from src.evaluation.prediction_alignment import validate_prediction_alignment
    expected = metadata[["Project", "Version", "Class"]].copy()
    expected["Actual"] = y.to_numpy()
    validate_prediction_alignment(reference, expected)

    root = output_dir / "feature_ablation"
    root.mkdir(parents=True, exist_ok=True)
    all_project, all_version, all_macro = [], [], []
    run_manifest = {
        "experiment": "Feature ablation with Leave-One-Project-Out",
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": "ds.csv",
        "observations": len(X),
        "projects": int(metadata["Project"].nunique()),
        "folds_per_configuration": int(metadata["Project"].nunique()),
        "environment": {"scikit_learn_version": sklearn.__version__},
        "model": {
            "name": "RandomForestClassifier",
            "n_estimators": N_ESTIMATORS,
            "max_features": MAX_FEATURES,
            "class_weight": CLASS_WEIGHT,
            "random_state": RANDOM_SEED,
            "n_jobs": N_JOBS,
        },
        "preprocessing": "Median imputation fitted independently on each training fold.",
        "aggregation": "Version metrics averaged within project; macro metrics equally weight projects.",
        "configurations": {},
    }

    for name, removed in configurations.items():
        predictions, folds, runtime_seconds = _build_ablation_predictions(
            X, y, metadata, removed
        )
        from src.evaluation.prediction_alignment import validate_prediction_alignment
        validate_prediction_alignment(reference, predictions)
        version, project, macro = _evaluate_variant(predictions)
        variant_dir = root / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(variant_dir / "predictions.csv", index=False)
        version.to_csv(variant_dir / "version_metrics.csv", index=False)
        project.to_csv(variant_dir / "project_metrics.csv", index=False)
        macro.to_csv(variant_dir / "macro_metrics.csv", index=False)
        (variant_dir / "metadata.json").write_text(json.dumps({
            "configuration": name,
            "removed_features": removed,
            "retained_features": [f for f in FEATURES if f not in removed],
            "number_of_features": len(FEATURES) - len(removed),
            "number_of_observations": len(predictions),
            "number_of_folds": len(folds),
            "runtime_seconds": runtime_seconds,
            "model": run_manifest["model"],
            "preprocessing": run_manifest["preprocessing"],
        }, indent=2), encoding="utf-8")
        p = project.copy()
        p.insert(0, "Configuration", name)
        v = version.copy()
        v.insert(0, "Configuration", name)
        m = macro.copy()
        m.insert(0, "Configuration", name)
        all_project.append(p)
        all_version.append(v)
        all_macro.append(m)
        run_manifest["configurations"][name] = {
            "removed_features": removed,
            "retained_features": [f for f in FEATURES if f not in removed],
            "runtime_seconds": runtime_seconds,
            "predictions": len(predictions),
            "folds": len(folds),
        }

    pd.concat(all_project, ignore_index=True).to_csv(root / "project_metrics_all.csv", index=False)
    pd.concat(all_version, ignore_index=True).to_csv(root / "version_metrics_all.csv", index=False)
    pd.concat(all_macro, ignore_index=True).to_csv(root / "macro_metrics_all.csv", index=False)
    (root / "metadata.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return root
