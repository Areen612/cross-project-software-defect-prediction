"""Dataset-size, target-balance, and approved-predictor missingness charts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from src.config import BUG_COUNT_COLUMN, INPUT_PATH, TARGET_COLUMN
from src.visualization.config import category_output_dir
from src.visualization.loaders import ArtifactError, load_dataset_frame, load_run_metadata
from src.visualization.styling import add_source_note, configure_plot_style, save_figure


def _dataset_inputs(
    experiment_dir: str | Path,
    dataset_path: str | Path | None,
) -> tuple[pd.DataFrame, list[str]]:
    metadata_path = Path(experiment_dir) / "metadata.json"
    metadata = load_run_metadata(experiment_dir)
    features = metadata.get("features", {}).get("predictors")
    target = metadata.get("target", {})
    if not isinstance(features, list) or not features or len(set(features)) != len(features):
        raise ArtifactError(f"{metadata_path} has no valid predictor list")
    expected_definition = f"{TARGET_COLUMN} = {BUG_COUNT_COLUMN} > 0"
    if target.get("column") != TARGET_COLUMN or target.get("definition") != expected_definition:
        raise ArtifactError(
            f"dataset charts support only the documented target rule: {expected_definition}"
        )
    path = Path(dataset_path) if dataset_path else INPUT_PATH
    frame = load_dataset_frame(path)
    required = {"Project", BUG_COUNT_COLUMN, *features}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ArtifactError(f"{path} is missing dataset columns: {missing}")
    if frame["Project"].isna().any():
        raise ArtifactError(f"{path} has missing Project values")
    bug_counts = pd.to_numeric(frame[BUG_COUNT_COLUMN], errors="coerce")
    if bug_counts.isna().any() or (bug_counts < 0).any():
        raise ArtifactError(f"{path} has missing or invalid {BUG_COUNT_COLUMN} values")
    frame = frame.copy()
    frame[TARGET_COLUMN] = bug_counts > 0
    return frame, features


def class_distribution_chart(
    frame: pd.DataFrame,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    configure_plot_style()
    counts = frame[TARGET_COLUMN].value_counts().reindex([False, True], fill_value=0)
    labels = ["Non-buggy", "Buggy"]
    values = [int(counts.loc[False]), int(counts.loc[True])]
    total = sum(values)
    colors = ["#999999", "#0072B2"]
    figure, axis = plt.subplots(figsize=(8.5, 3.4), constrained_layout=True)
    bars = axis.barh(labels, values, color=colors, height=0.56)
    axis.set_xlim(0, max(values) * 1.16)
    axis.set_xlabel("Observations")
    axis.set_title("Class balance from bugs > 0")
    for bar, count in zip(bars, values):
        percent = count / total if total else 0.0
        axis.text(
            bar.get_width() + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}  ({percent:.1%})",
            va="center",
            fontsize=10,
        )
    add_source_note(figure, "Target is derived in memory using the saved run definition; the CSV is unchanged.")
    return save_figure(
        figure,
        category_output_dir(output_dir, "dataset") / "class_distribution",
        formats,
    )


def project_size_chart(
    frame: pd.DataFrame,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    configure_plot_style()
    sizes = frame.groupby("Project", dropna=False).size().sort_values()
    figure, axis = plt.subplots(
        figsize=(10.5, max(8.5, len(sizes) * 0.26)), constrained_layout=True
    )
    axis.barh(sizes.index.astype(str), sizes.values, color="#4C78A8", height=0.72)
    axis.set_xscale("log")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.set_xlabel("Observations per project (log scale)")
    axis.set_ylabel("Project")
    axis.set_title("Dataset size by project")
    add_source_note(figure, f"{len(sizes)} projects; counts are source observations.")
    return save_figure(
        figure,
        category_output_dir(output_dir, "dataset") / "project_size_distribution",
        formats,
    )


def feature_missingness_chart(
    frame: pd.DataFrame,
    features: list[str],
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    configure_plot_style()
    missing = frame[features].isna().sum().rename("Missing_Count").to_frame()
    missing["Missing_Percent"] = missing["Missing_Count"] / len(frame)
    missing = missing.sort_values(["Missing_Percent", "Missing_Count"])
    figure, axis = plt.subplots(
        figsize=(9.5, max(6.5, len(missing) * 0.31)), constrained_layout=True
    )
    axis.barh(missing.index, missing["Missing_Percent"] * 100, color="#72A0C1", height=0.72)
    xmax = max(float((missing["Missing_Percent"] * 100).max()) * 1.6, 0.2)
    axis.set_xlim(0, xmax)
    axis.set_xlabel("Missing source values (%)")
    axis.set_ylabel("Approved predictor")
    axis.set_title("Feature missingness")
    for y, (_, row) in enumerate(missing.iterrows()):
        axis.text(
            row["Missing_Percent"] * 100 + xmax * 0.02,
            y,
            f"{int(row['Missing_Count']):,}",
            va="center",
            fontsize=8,
        )
    add_source_note(figure, "Only predictors recorded in the run metadata are shown; ‘-’ is read as missing.")
    return save_figure(
        figure,
        category_output_dir(output_dir, "dataset") / "feature_missingness",
        formats,
    )


def generate_dataset_figures(
    experiment_dir: str | Path,
    output_dir: str | Path,
    dataset_path: str | Path | None = None,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    frame, features = _dataset_inputs(experiment_dir, dataset_path)
    return [
        *class_distribution_chart(frame, output_dir, formats),
        *project_size_chart(frame, output_dir, formats),
        *feature_missingness_chart(frame, features, output_dir, formats),
    ]
