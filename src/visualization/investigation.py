"""Charts for quantified findings documented in the read-only investigation report."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.config import category_output_dir
from src.visualization.loaders import ArtifactError
from src.visualization.styling import add_source_note, configure_plot_style, save_figure


def _integer(text: str) -> int:
    return int(text.replace(",", ""))


def parse_investigation_report(path: str | Path) -> dict:
    """Extract only explicitly quantified values from the existing Markdown report."""
    source = Path(path)
    if not source.is_file():
        raise ArtifactError(f"investigation report not found: {source}")
    try:
        report = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ArtifactError(f"could not read investigation report {source}: {exc}") from exc

    associations = []
    for line in report.splitlines():
        match = re.match(
            r"\|\s*`([^`]+)`\s*\|\s*(-?\d+(?:\.\d+)?)\s*\|\s*(-?\d+(?:\.\d+)?)\s*\|",
            line,
        )
        if match:
            feature, auc, rho = match.groups()
            associations.append({
                "Feature": feature,
                "Directional_AUC": float(auc),
                "Spearman_Rho": float(rho),
            })
    if associations:
        association_frame = pd.DataFrame(associations)
        if association_frame["Feature"].duplicated().any():
            raise ArtifactError(f"{source} has duplicate feature-association rows")
        if (
            (association_frame["Directional_AUC"] < 0.5).any()
            or (association_frame["Directional_AUC"] > 1).any()
            or (association_frame["Spearman_Rho"].abs() > 1).any()
        ):
            raise ArtifactError(f"{source} has feature-association values outside their valid ranges")
    else:
        association_frame = pd.DataFrame(
            columns=["Feature", "Directional_AUC", "Spearman_Rho"]
        )

    overlap = re.search(
        r"Exact matches on all 18 selected feature values occur across projects:\s*"
        r"\*\*([\d,]+) feature-vector groups\*\* span multiple projects and cover "
        r"\*\*([\d,]+) rows\*\*",
        report,
    )
    missing_versions = re.search(r"\*\*([\d,]+) rows\*\* have no `Version` value", report)
    repeated_keys = re.search(
        r"\*\*([\d,]+) repeated `\(Project, Version, Class\)` keys\*\*, "
        r"accounting for \*\*([\d,]+) row occurrences\*\*",
        report,
    )
    duplicate_rows = re.search(
        r"\*\*([\d,]+) rows\*\* belong to exact duplicate groups", report
    )
    return {
        "associations": association_frame,
        "overlap_groups": _integer(overlap.group(1)) if overlap else None,
        "overlap_rows": _integer(overlap.group(2)) if overlap else None,
        "missing_version_rows": _integer(missing_versions.group(1)) if missing_versions else None,
        "repeated_key_count": _integer(repeated_keys.group(1)) if repeated_keys else None,
        "repeated_key_rows": _integer(repeated_keys.group(2)) if repeated_keys else None,
        "exact_duplicate_rows": _integer(duplicate_rows.group(1)) if duplicate_rows else None,
    }


def feature_association_chart(
    associations: pd.DataFrame,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    configure_plot_style()
    data = associations.sort_values("Directional_AUC").copy()
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 8.5), sharey=True, constrained_layout=True)
    axes[0].barh(data["Feature"], data["Directional_AUC"], color="#4C78A8", height=0.72)
    axes[0].axvline(0.5, color="#555555", linewidth=1, linestyle="--")
    axes[0].set_xlim(0.5, 1.02)
    axes[0].set_xlabel("Directional AUC")
    axes[0].set_ylabel("Predictor")
    axes[0].set_title("Directional AUC")
    axes[1].barh(data["Feature"], data["Spearman_Rho"], color="#72A0C1", height=0.72)
    axes[1].axvline(0, color="#555555", linewidth=1)
    axes[1].set_xlim(-1.02, 1.02)
    axes[1].set_xlabel("Spearman correlation")
    axes[1].set_ylabel("")
    axes[1].set_title("Spearman ρ")
    figure.suptitle("Documented predictor–target associations", fontweight="bold")
    add_source_note(
        figure,
        "Descriptive associations from the investigation report; they do not establish causation or data leakage.",
    )
    return save_figure(
        figure,
        category_output_dir(output_dir, "investigation") / "feature_associations",
        formats,
    )


def cross_project_overlap_chart(
    overlap_rows: int,
    overlap_groups: int,
    total_observations: int,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    if total_observations <= 0 or overlap_rows < 0 or overlap_rows > total_observations:
        raise ArtifactError("investigation overlap counts do not fit the saved observation total")
    configure_plot_style()
    unmatched = total_observations - overlap_rows
    matched_pct = overlap_rows / total_observations
    figure, axis = plt.subplots(figsize=(10, 4.4))
    axis.barh(["Dataset observations"], [overlap_rows], color="#0072B2", height=0.48, label="In cross-project matches")
    axis.barh(
        ["Dataset observations"], [unmatched], left=[overlap_rows],
        color="#D9DEE5", height=0.48, label="Not in a documented cross-project match",
    )
    axis.text(overlap_rows / 2, 0, f"{overlap_rows:,}\n{matched_pct:.1%}", color="white", ha="center", va="center", fontweight="bold")
    if unmatched:
        axis.text(overlap_rows + unmatched / 2, 0, f"{unmatched:,}\n{1 - matched_pct:.1%}", color="#333333", ha="center", va="center")
    axis.set_xlim(0, total_observations)
    axis.set_xlabel("Observations")
    axis.set_title("Rows in documented cross-project feature-vector matches")
    figure.subplots_adjust(left=0.18, right=0.98, top=0.84, bottom=0.38)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=2,
    )
    add_source_note(
        figure,
        f"Report: {overlap_groups:,} distinct exact 18-feature groups span multiple projects. Overlap is descriptive, not proof of leakage.",
        y=0.055,
        va="bottom",
    )
    return save_figure(
        figure,
        category_output_dir(output_dir, "investigation") / "cross_project_overlap",
        formats,
    )


def dataset_integrity_chart(
    findings: dict,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> list[Path]:
    rows = [
        ("Rows in cross-project exact feature matches", findings.get("overlap_rows")),
        ("Rows without a Version value", findings.get("missing_version_rows")),
        ("Rows in repeated project/version/class keys", findings.get("repeated_key_rows")),
        ("Rows in exact duplicate groups", findings.get("exact_duplicate_rows")),
    ]
    data = pd.DataFrame(
        [(label, value) for label, value in rows if value is not None],
        columns=["Finding", "Rows"],
    ).sort_values("Rows")
    if data.empty:
        raise ArtifactError("the investigation report has no quantified row-count findings")
    configure_plot_style()
    figure, axis = plt.subplots(
        figsize=(11, max(4.2, len(data) * 0.8)), constrained_layout=True
    )
    bars = axis.barh(data["Finding"], data["Rows"], color="#7A9E7E", height=0.62)
    axis.set_xlabel("Rows")
    axis.set_title("Quantified dataset-integrity findings")
    for bar, value in zip(bars, data["Rows"]):
        axis.text(value + max(data["Rows"]) * 0.015, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center")
    add_source_note(figure, "Counts come from the investigation report and may overlap; do not add them together.")
    return save_figure(
        figure,
        category_output_dir(output_dir, "investigation") / "dataset_integrity_counts",
        formats,
    )


def generate_investigation_figures(
    report_path: str | Path,
    total_observations: int,
    output_dir: str | Path,
    formats: str | Iterable[str] = "png",
) -> tuple[list[Path], list[str]]:
    findings = parse_investigation_report(report_path)
    generated: list[Path] = []
    skipped: list[str] = []
    associations = findings["associations"]
    if not associations.empty:
        generated.extend(feature_association_chart(associations, output_dir, formats))
    else:
        skipped.append("feature associations: no quantified association table in the report")
    if findings["overlap_rows"] is not None and findings["overlap_groups"] is not None:
        try:
            generated.extend(cross_project_overlap_chart(
                findings["overlap_rows"], findings["overlap_groups"],
                total_observations, output_dir, formats,
            ))
        except ArtifactError as exc:
            skipped.append(f"cross-project overlap: {exc}")
    else:
        skipped.append("cross-project overlap: no quantified overlap summary in the report")
    integrity_findings = dict(findings)
    invalid_overlap = (
        findings["overlap_rows"] is not None
        and (findings["overlap_rows"] < 0 or findings["overlap_rows"] > total_observations)
    )
    if invalid_overlap:
        integrity_findings["overlap_rows"] = None
    try:
        generated.extend(dataset_integrity_chart(integrity_findings, output_dir, formats))
    except ArtifactError as exc:
        skipped.append(f"dataset integrity: {exc}")
    skipped.append(
        "overlap-versus-unseen performance: unavailable; no saved group-level metric table or stable prediction row ID exists"
    )
    return generated, skipped
