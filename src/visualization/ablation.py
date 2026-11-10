"""Charts for saved feature-ablation metrics and their full-feature baseline."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.config import MODEL_COLORS, category_output_dir
from src.visualization.loaders import AblationArtifacts, load_ablation_artifacts
from src.visualization.styling import add_source_note, configure_plot_style, save_figure


def _config_order(artifacts: AblationArtifacts) -> list[str]:
    return list(artifacts.metadata["configurations"])


def _config_label(artifacts: AblationArtifacts, name: str) -> str:
    removed = artifacts.metadata["configurations"][name].get("removed_features", [])
    detail = ", ".join(removed) if removed else "none"
    return f"{name}\nremoved: {detail}"


def _macro_chart(
    artifacts: AblationArtifacts,
    output_dir: str | Path,
    formats: str | Iterable[str],
    metric: str,
    filename: str,
    title: str,
    ylabel: str,
    note: str,
) -> list[Path]:
    configure_plot_style()
    order = _config_order(artifacts)
    data = artifacts.macro.set_index("Configuration").loc[order].reset_index()
    data["Display"] = data["Configuration"].map(lambda name: _config_label(artifacts, name))
    data = data.dropna(subset=[metric])
    figure_height = max(4.8, len(data) * 0.72)
    figure, axis = plt.subplots(figsize=(11.5, figure_height), constrained_layout=True)
    color = MODEL_COLORS["RandomForest"]
    axis.barh(data["Display"], data[metric], color=color, height=0.66)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.08)
    axis.set_xlabel(ylabel)
    axis.set_ylabel("")
    axis.set_title(title)
    for y, value in enumerate(data[metric].tolist()):
        axis.text(value + 0.012, y, f"{value:.4f}", va="center", fontsize=9)
    add_source_note(figure, note)
    return save_figure(
        figure,
        category_output_dir(output_dir, "ablation") / filename,
        formats,
    )


def feature_ablation_mcc_chart(artifacts, output_dir, formats="png") -> list[Path]:
    return _macro_chart(
        artifacts,
        output_dir,
        formats,
        "MCC",
        "feature_ablation_mcc",
        "Feature-ablation MCC",
        "Macro average of version-level MCC within each project",
        "Each project contributes equally after averaging its informative version MCC values.",
    )


def feature_ablation_pooled_mcc_chart(artifacts, output_dir, formats="png") -> list[Path]:
    return _macro_chart(
        artifacts,
        output_dir,
        formats,
        "Pooled_MCC",
        "feature_ablation_pooled_mcc",
        "Feature-ablation pooled MCC",
        "Macro average of project-level pooled MCC",
        "Separate aggregation: confusion counts are pooled within each project, then projects are averaged.",
    )


def feature_ablation_f1_chart(artifacts, output_dir, formats="png") -> list[Path]:
    return _macro_chart(
        artifacts,
        output_dir,
        formats,
        "F1",
        "feature_ablation_f1",
        "Feature-ablation F1",
        "Macro average of version-level F1 within each project",
        "Each project contributes equally after averaging its version-level F1 values.",
    )


def feature_ablation_delta_chart(
    artifacts: AblationArtifacts,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    configure_plot_style()
    order = _config_order(artifacts)
    baseline = "baseline_all_features"
    if baseline not in order:
        raise ValueError("ablation metadata is missing baseline_all_features")
    values = artifacts.macro.set_index("Configuration")["MCC"]
    baseline_value = float(values.loc[baseline])
    configs = [name for name in order if name != baseline]
    deltas = pd.DataFrame({
        "Configuration": configs,
        "Delta": [float(values.loc[name]) - baseline_value for name in configs],
    })
    deltas["Display"] = deltas["Configuration"].map(
        lambda name: _config_label(artifacts, name)
    )
    figure, axis = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    axis.barh(deltas["Display"], deltas["Delta"], color="#4C78A8", height=0.62)
    axis.axvline(0, color="#333333", linewidth=1)
    axis.invert_yaxis()
    extent = max(float(deltas["Delta"].abs().max()) * 1.35, 0.002)
    axis.set_xlim(-extent, extent)
    axis.set_xlabel("Change in macro MCC from baseline")
    axis.set_ylabel("")
    axis.set_title("Feature-ablation change from the full-feature baseline")
    for y, value in enumerate(deltas["Delta"].tolist()):
        offset = extent * 0.025 if value >= 0 else -extent * 0.025
        axis.text(value + offset, y, f"{value:+.4f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    add_source_note(
        figure,
        "Changes use the same version-level MCC then project-macro aggregation as feature_ablation_mcc.png.",
    )
    return save_figure(
        figure,
        category_output_dir(output_dir, "ablation") / "feature_ablation_delta",
        formats,
    )


def feature_ablation_project_distribution_chart(
    artifacts: AblationArtifacts,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    configure_plot_style()
    order = _config_order(artifacts)
    data = artifacts.projects.dropna(subset=["MCC"]).copy()
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sns.boxplot(
        data=data,
        x="Configuration",
        y="MCC",
        order=order,
        color="#A8C5DA",
        showfliers=False,
        ax=axis,
    )
    sns.stripplot(
        data=data,
        x="Configuration",
        y="MCC",
        order=order,
        color="#222222",
        size=3,
        alpha=0.45,
        jitter=0.18,
        ax=axis,
    )
    axis.set_xticks(
        range(len(order)),
        labels=[fill(name, 20) for name in order],
        rotation=0,
        ha="center",
        fontsize=8,
    )
    axis.set_ylim(-1.03, 1.03)
    axis.set_xlabel("Saved configuration name")
    axis.set_ylabel("Per-project mean of informative version MCC")
    axis.set_title("Project-level variation across ablation configurations")
    add_source_note(
        figure,
        "Each dot is one project; project MCC is the mean of informative version-level MCC values.",
    )
    return save_figure(
        figure,
        category_output_dir(output_dir, "ablation") / "feature_ablation_project_mcc_distribution",
        formats,
    )


def generate_ablation_figures(
    experiment_dir: str | Path,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
    artifacts: AblationArtifacts | None = None,
) -> list[Path]:
    loaded = artifacts or load_ablation_artifacts(experiment_dir)
    return [
        *feature_ablation_mcc_chart(loaded, output_dir, formats),
        *feature_ablation_delta_chart(loaded, output_dir, formats),
        *feature_ablation_pooled_mcc_chart(loaded, output_dir, formats),
        *feature_ablation_f1_chart(loaded, output_dir, formats),
        *feature_ablation_project_distribution_chart(loaded, output_dir, formats),
    ]
