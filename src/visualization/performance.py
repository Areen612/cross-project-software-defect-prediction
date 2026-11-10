"""Overall performance charts from the saved evaluation workbook."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from src.visualization.config import (
    MODEL_COLORS,
    MODEL_LABELS,
    WORKBOOK_MODEL_KEYS,
    category_output_dir,
)
from src.visualization.loaders import (
    EvaluationArtifacts,
    load_evaluation_artifacts,
)
from src.visualization.styling import add_source_note, configure_plot_style, save_figure


_PRIMARY_METRICS = ["Accuracy", "Precision", "Recall", "F1"]
_WEIGHTED_METRICS = [
    "Support-weighted Accuracy",
    "Support-weighted Precision",
    "Support-weighted Recall",
    "Support-weighted F1",
]


def _macro_rows(comparison: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    data = comparison[
        comparison["Evaluation_Level"].eq("Macro average")
        & comparison["Metric"].isin(metrics)
    ].copy()
    data["Model_Key"] = data["Model"].map(WORKBOOK_MODEL_KEYS)
    data["Model_Label"] = data["Model_Key"].map(MODEL_LABELS)
    data["Metric"] = data["Metric"].replace(
        dict(zip(_WEIGHTED_METRICS, _PRIMARY_METRICS))
    )
    data = data.dropna(subset=["Value", "Model_Label"])
    if data.empty:
        raise ValueError(f"no available macro values for {metrics}")
    return data


def _grouped_macro_chart(
    comparison: pd.DataFrame,
    output_dir: str | Path,
    formats: str | Iterable[str],
    *,
    metrics: list[str],
    filename: str,
    title: str,
    ylabel: str,
    note: str,
) -> list[Path]:
    configure_plot_style()
    data = _macro_rows(comparison, metrics)
    ordered_labels = [MODEL_LABELS[key] for key in MODEL_LABELS]
    data["Metric"] = pd.Categorical(data["Metric"], _PRIMARY_METRICS, ordered=True)
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    sns.barplot(
        data=data,
        x="Metric",
        y="Value",
        hue="Model_Label",
        hue_order=ordered_labels,
        palette={MODEL_LABELS[key]: value for key, value in MODEL_COLORS.items()},
        errorbar=None,
        ax=axis,
    )
    axis.set_ylim(0, 1.08)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_xlabel("")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )
    figure.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.18)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    add_source_note(figure, note)
    return save_figure(
        figure,
        category_output_dir(output_dir, "performance") / filename,
        formats,
    )


def overall_performance_chart(
    comparison: pd.DataFrame,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    """Plot the primary project-macro metrics defined by the workbook."""
    return _grouped_macro_chart(
        comparison,
        output_dir,
        formats,
        metrics=_PRIMARY_METRICS,
        filename="model_performance_comparison",
        title="LOPO primary macro metrics",
        ylabel="Macro-average score across held-out projects",
        note=(
            "Version metrics are averaged within each project; each project contributes equally to the macro mean."
        ),
    )


def support_weighted_performance_chart(
    comparison: pd.DataFrame,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    """Show supplementary support-weighted project metrics separately."""
    return _grouped_macro_chart(
        comparison,
        output_dir,
        formats,
        metrics=_WEIGHTED_METRICS,
        filename="support_weighted_model_performance",
        title="Supplementary support-weighted metrics",
        ylabel="Project macro-average score",
        note=(
            "Version scores are support-weighted within each project; projects then contribute equally."
        ),
    )


def _mcc_comparison_chart(
    comparison: pd.DataFrame,
    output_dir: str | Path,
    formats: str | Iterable[str],
    *,
    metric: str,
    filename: str,
    title: str,
    xlabel: str,
    note: str,
) -> list[Path]:
    configure_plot_style()
    data = comparison[
        comparison["Evaluation_Level"].eq("Macro average")
        & comparison["Metric"].eq(metric)
    ].copy()
    data["Model_Key"] = data["Model"].map(WORKBOOK_MODEL_KEYS)
    data["Model_Label"] = data["Model_Key"].map(MODEL_LABELS)
    if data.empty:
        raise ValueError(f"no macro rows are available for {metric}")

    order = [MODEL_LABELS[key] for key in MODEL_LABELS]
    values_by_model = data.set_index("Model_Label")["Value"].to_dict()
    status_by_model = data.set_index("Model_Label")["Status"].to_dict()
    figure, axis = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    positions = list(range(len(order)))
    for position, label in zip(positions, order):
        value = values_by_model.get(label)
        if pd.notna(value):
            axis.barh(
                [position], [float(value)],
                color=MODEL_COLORS[
                    next(key for key, display in MODEL_LABELS.items() if display == label)
                ],
                height=0.48,
            )
            x = float(value) + (0.02 if value >= 0 else -0.02)
            axis.text(
                x,
                position,
                f"{float(value):.4f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9,
            )
        else:
            status = status_by_model.get(label)
            reason = {
                "No informative groups": "undefined (no informative projects)",
                "Pending": "pending",
            }.get(status, "unavailable")
            axis.text(0.02, position, reason, va="center", fontsize=9)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_xlim(-1.0, 1.2)
    axis.set_ylim(-0.65, len(order) - 0.35)
    axis.invert_yaxis()
    axis.set_yticks(positions, order)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    add_source_note(figure, note)
    return save_figure(
        figure,
        category_output_dir(output_dir, "performance") / filename,
        formats,
    )


def mcc_comparison_chart(
    comparison: pd.DataFrame,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    """Show primary MCC, averaged from informative versions and projects."""
    return _mcc_comparison_chart(
        comparison,
        output_dir,
        formats,
        metric="MCC",
        filename="model_mcc_comparison",
        title="Primary macro MCC",
        xlabel="Macro mean of informative project MCC values",
        note=(
            "Each project averages its informative version MCC values first; projects then contribute equally."
        ),
    )


def pooled_mcc_comparison_chart(
    comparison: pd.DataFrame,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    """Plot the separate macro mean of project-level pooled MCC values."""
    return _mcc_comparison_chart(
        comparison,
        output_dir,
        formats,
        metric="Pooled MCC",
        filename="pooled_mcc_comparison",
        title="Supplementary pooled MCC",
        xlabel="Macro mean of informative project-pooled MCC values",
        note=(
            "Confusion counts are combined within each project before MCC; informative projects are then averaged equally."
        ),
    )


def generate_performance_figures(
    experiment_dir: str | Path,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
    artifacts: EvaluationArtifacts | None = None,
) -> list[Path]:
    loaded = artifacts or load_evaluation_artifacts(experiment_dir)
    return [
        *overall_performance_chart(loaded.comparison, output_dir, formats),
        *mcc_comparison_chart(loaded.comparison, output_dir, formats),
        *pooled_mcc_comparison_chart(loaded.comparison, output_dir, formats),
        *support_weighted_performance_chart(loaded.comparison, output_dir, formats),
    ]
