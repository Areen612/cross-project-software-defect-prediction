"""Version-level robustness views from saved Random Forest and Dummy results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.config import category_output_dir
from src.visualization.loaders import RobustnessArtifacts, load_robustness_artifacts
from src.visualization.styling import add_source_note, configure_plot_style, save_figure


_METRICS = ["Accuracy", "F1", "Specificity", "Balanced_Accuracy"]


def version_metric_delta_chart(
    versions: pd.DataFrame,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    """Plot paired Random Forest-minus-Dummy changes by project-version group."""
    configure_plot_style()
    key_columns = ["Project", "Version"]
    charts = []
    for metric in _METRICS:
        wide = versions.pivot(index=key_columns, columns="Model", values=metric)
        if not {"RandomForest", "DummyClassifier"}.issubset(wide.columns):
            continue
        delta = (wide["RandomForest"] - wide["DummyClassifier"]).dropna()
        if delta.empty:
            continue
        charts.append((metric, delta.reset_index(drop=True)))
    if not charts:
        raise ValueError("no paired version metrics are available for robustness charts")

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        if index >= len(charts):
            axis.set_visible(False)
            continue
        metric, delta = charts[index]
        values = delta.to_numpy(dtype=float)
        jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.array([0.0])
        axis.axvline(0, color="#555555", linewidth=1, linestyle="--", zorder=1)
        axis.scatter(values, jitter, s=16, alpha=0.5, color="#4C566A", zorder=2)
        median = float(np.median(values))
        axis.axvline(median, color="#0072B2", linewidth=2, zorder=3)
        axis.set_yticks([])
        axis.set_xlabel("Random Forest minus Dummy")
        axis.set_title(f"{metric} (n={len(values)} groups)")
        span = max(float(np.max(np.abs(values))), 0.01)
        axis.set_xlim(-span * 1.15, span * 1.15)
    figure.suptitle("Random Forest vs Dummy by project-version group", fontweight="bold")
    add_source_note(
        figure,
        "Paired differences from saved predictions; positive values favor Random Forest. These are model comparisons, not configuration variants or repeated runs.",
    )
    return save_figure(
        figure,
        category_output_dir(output_dir, "robustness") / "robustness_version_metric_deltas",
        formats,
    )


def generate_robustness_figures(
    experiment_dir: str | Path,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
    artifacts: RobustnessArtifacts | None = None,
) -> list[Path]:
    loaded = artifacts or load_robustness_artifacts(experiment_dir)
    return version_metric_delta_chart(loaded.versions, output_dir, formats)
