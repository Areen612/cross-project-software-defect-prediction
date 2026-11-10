"""Consistent, publication-oriented style and figure export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization.config import FIGURE_DPI, SUPPORTED_FORMATS


def configure_plot_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font_scale=0.95,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.dpi": 120,
            "savefig.dpi": FIGURE_DPI,
            "font.family": "DejaVu Sans",
        },
    )


def save_figure(
    figure: plt.Figure,
    output_base: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    """Save and close a figure in the requested raster/vector formats."""
    if isinstance(formats, str):
        selected = [formats]
    else:
        selected = list(formats)
    selected = list(dict.fromkeys(fmt.lower().lstrip(".") for fmt in selected))
    invalid = sorted(set(selected) - SUPPORTED_FORMATS)
    if invalid or not selected:
        plt.close(figure)
        raise ValueError(
            f"formats must be a non-empty subset of {sorted(SUPPORTED_FORMATS)}; "
            f"received {selected}"
        )
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in selected:
        path = base.with_suffix(f".{extension}")
        figure.savefig(
            path,
            format=extension,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            facecolor="white",
        )
        paths.append(path)
    plt.close(figure)
    return paths


def add_source_note(
    figure: plt.Figure,
    text: str,
    *,
    y: float = -0.035,
    va: str = "top",
) -> None:
    # Place notes below axes so they never collide with x-axis labels or legends.
    figure.text(0.01, y, text, ha="left", va=va, fontsize=8, color="#555555")
