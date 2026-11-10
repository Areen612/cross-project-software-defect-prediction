from __future__ import annotations

import json

import pandas as pd

from src.visualization.ablation import generate_ablation_figures
from src.visualization.dataset import generate_dataset_figures
from src.visualization.investigation import generate_investigation_figures, parse_investigation_report
from src.visualization.loaders import (
    load_ablation_artifacts,
    load_evaluation_artifacts,
    load_robustness_artifacts,
)
from src.visualization.performance import generate_performance_figures
from src.visualization.project_analysis import generate_project_figures
from src.visualization.robustness import generate_robustness_figures
from src.visualization.styling import save_figure

import matplotlib.pyplot as plt


def test_save_figure_supports_png_and_svg(tmp_path):
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    paths = save_figure(figure, tmp_path / "chart", formats=("png", "svg"))
    assert {path.suffix for path in paths} == {".png", ".svg"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_saved_robustness_artifacts_generate_figures(robustness_run, tmp_path):
    source_files = sorted((robustness_run / "robustness").glob("*.csv"))
    before = {path: path.read_bytes() for path in source_files}
    evaluation_path = robustness_run / "evaluation_workbook.xlsx"
    evaluation_before = evaluation_path.read_bytes()
    evaluation = load_evaluation_artifacts(robustness_run)
    performance = generate_performance_figures(robustness_run, tmp_path / "figs", artifacts=evaluation)
    projects = generate_project_figures(robustness_run, tmp_path / "figs", artifacts=evaluation)
    robustness_data = load_robustness_artifacts(robustness_run)
    robustness = generate_robustness_figures(robustness_run, tmp_path / "figs", artifacts=robustness_data)
    assert len(performance) == 4
    assert len(projects) == 3
    assert len(robustness) == 1
    assert "pooled_mcc_comparison.png" in {path.name for path in performance}
    assert "support_weighted_model_performance.png" in {path.name for path in performance}
    assert all(path.is_file() and path.stat().st_size > 0 for path in performance + projects + robustness)
    assert before == {path: path.read_bytes() for path in source_files}
    assert evaluation_before == evaluation_path.read_bytes()


def test_ablation_figures_keep_primary_and_pooled_mcc_separate(ablation_run, tmp_path):
    artifacts = load_ablation_artifacts(ablation_run)
    paths = generate_ablation_figures(ablation_run, tmp_path / "figs", artifacts=artifacts)
    names = {path.name for path in paths}
    assert "feature_ablation_mcc.png" in names
    assert "feature_ablation_pooled_mcc.png" in names
    assert "feature_ablation_delta.png" in names
    assert len(paths) == 5


def test_dataset_figures_use_run_feature_list_and_do_not_modify_dataset(
    robustness_run, dataset_csv, tmp_path
):
    before = dataset_csv.read_bytes()
    paths = generate_dataset_figures(
        robustness_run,
        tmp_path / "figs",
        dataset_path=dataset_csv,
    )
    assert {path.name for path in paths} == {
        "class_distribution.png",
        "project_size_distribution.png",
        "feature_missingness.png",
    }
    assert dataset_csv.read_bytes() == before


def test_investigation_report_only_generates_quantified_charts(tmp_path):
    report = tmp_path / "investigation.md"
    report.write_text(
        "## 3. Predictor-target relationships\n"
        "| Feature | Directional AUC | Spearman rho |\n"
        "|---|---:|---:|\n"
        "| `loc` | 0.9474 | 0.4304 |\n"
        "| `dam` | 0.7647 | -0.2549 |\n"
        "## 4. Dataset duplicates, repeated keys, and cross-project overlap\n"
        "- **6 rows** have no `Version` value.\n"
        "- **2 repeated `(Project, Version, Class)` keys**, accounting for **3 row occurrences**.\n"
        "- **1 rows** belong to exact duplicate groups after excluding the index-like column.\n"
        "- Exact matches on all 18 selected feature values occur across projects: **4 feature-vector groups** span multiple projects and cover **5 rows**.\n",
        encoding="utf-8",
    )
    findings = parse_investigation_report(report)
    assert len(findings["associations"]) == 2
    assert findings["overlap_groups"] == 4
    assert findings["overlap_rows"] == 5
    generated, skipped = generate_investigation_figures(
        report, total_observations=10, output_dir=tmp_path / "figs"
    )
    assert len(generated) == 3
    assert any("overlap-versus-unseen performance: unavailable" in item for item in skipped)


def test_investigation_overlap_count_must_fit_dataset(tmp_path):
    report = tmp_path / "investigation.md"
    report.write_text(
        "Exact matches on all 18 selected feature values occur across projects: "
        "**2 feature-vector groups** span multiple projects and cover **12 rows**.",
        encoding="utf-8",
    )
    generated, skipped = generate_investigation_figures(
        report, total_observations=10, output_dir=tmp_path / "figs"
    )
    assert not generated
    assert any("overlap counts do not fit" in item for item in skipped)
