"""Charts showing project-level performance and its project-to-project spread."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from src.visualization.config import MODEL_COLORS, MODEL_LABELS, category_output_dir
from src.visualization.loaders import (
    EvaluationArtifacts,
    evaluation_project_metrics,
    load_evaluation_artifacts,
)
from src.visualization.styling import add_source_note, configure_plot_style, save_figure


def project_model_comparison_chart(
    projects: pd.DataFrame,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    """Compare saved Random Forest and Dummy scores for every project."""
    configure_plot_style()
    data = projects[projects["Model"].isin(MODEL_LABELS)].copy()
    data["Model_Label"] = data["Model"].map(MODEL_LABELS)
    rf_order = (
        data[data["Model"].eq("RandomForest")]
        .sort_values("Accuracy")["Project"].tolist()
    )
    figure_height = max(12, len(rf_order) * 0.34)
    figure, axes = plt.subplots(
        1, 2, figsize=(15, figure_height), sharey=True
    )
    for axis, metric in zip(axes, ("Accuracy", "F1")):
        sns.barplot(
            data=data,
            y="Project",
            x=metric,
            hue="Model_Label",
            hue_order=[MODEL_LABELS[name] for name in MODEL_LABELS],
            order=rf_order,
            palette={MODEL_LABELS[key]: value for key, value in MODEL_COLORS.items()},
            errorbar=None,
            ax=axis,
        )
        axis.set_xlim(0, 1.07)
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.set_title(metric)
        axis.set_xlabel(metric)
        axis.set_ylabel("Held-out project" if axis is axes[0] else "")
        axis.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Model",
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=2,
    )
    figure.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.15, wspace=0.1)
    add_source_note(
        figure,
        "Each project metric is an unweighted mean of its version metrics; rows are not individual observations.",
    )
    return save_figure(
        figure,
        category_output_dir(output_dir, "projects") / "project_model_comparison",
        formats,
    )


def project_mcc_chart(
    projects: pd.DataFrame,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    """Show MCC for each informative held-out project."""
    configure_plot_style()
    rf = projects[projects["Model"].eq("RandomForest")].dropna(subset=["MCC"]).copy()
    rf = rf.sort_values("MCC")
    figure, axis = plt.subplots(
        figsize=(10.5, max(8.5, len(rf) * 0.27)), constrained_layout=True
    )
    axis.barh(rf["Project"], rf["MCC"], color=MODEL_COLORS["RandomForest"], height=0.72)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_xlim(-1, 1.04)
    axis.set_xlabel("MCC (higher is better)")
    axis.set_ylabel("Held-out project")
    axis.set_title("Random Forest MCC by held-out project")
    missing = int(projects.loc[projects["Model"].eq("RandomForest"), "MCC"].isna().sum())
    add_source_note(
        figure,
        f"MCC is informative for {len(rf)} projects; {missing} undefined or non-informative project values are not shown.",
    )
    return save_figure(
        figure,
        category_output_dir(output_dir, "projects") / "project_mcc_comparison",
        formats,
    )


def project_metric_distribution_chart(
    projects: pd.DataFrame,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    """Show project-to-project spread for scores with meaningful dummy values."""
    configure_plot_style()
    data = projects[projects["Model"].isin(MODEL_LABELS)].copy()
    data["Model_Label"] = data["Model"].map(MODEL_LABELS)
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics):
        metric_data = data.dropna(subset=[metric])
        sns.boxplot(
            data=metric_data,
            x="Model_Label",
            y=metric,
            order=[MODEL_LABELS[name] for name in MODEL_LABELS],
            hue="Model_Label",
            hue_order=[MODEL_LABELS[name] for name in MODEL_LABELS],
            palette={MODEL_LABELS[key]: value for key, value in MODEL_COLORS.items()},
            legend=False,
            showfliers=False,
            width=0.52,
            ax=axis,
        )
        sns.stripplot(
            data=metric_data,
            x="Model_Label",
            y=metric,
            order=[MODEL_LABELS[name] for name in MODEL_LABELS],
            color="#292929",
            size=3,
            alpha=0.45,
            jitter=0.16,
            ax=axis,
        )
        axis.set_ylim(-0.03, 1.03)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_title(f"{metric} ({len(metric_data)} projects)")
        axis.set_xlabel("")
        axis.set_ylabel("Score")
    figure.suptitle("Project-to-project performance spread", fontweight="bold")
    add_source_note(
        figure,
        "Each dot is one held-out project; its primary metric is an unweighted mean across informative versions.",
    )
    return save_figure(
        figure,
        category_output_dir(output_dir, "projects") / "project_performance_distribution",
        formats,
    )


def generate_project_figures(
    experiment_dir: str | Path,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
    artifacts: EvaluationArtifacts | None = None,
) -> list[Path]:
    loaded = artifacts or load_evaluation_artifacts(experiment_dir)
    projects = evaluation_project_metrics(loaded)
    return [
        *project_mcc_chart(projects, output_dir, formats),
        *project_model_comparison_chart(projects, output_dir, formats),
        *project_metric_distribution_chart(projects, output_dir, formats),
    ]
