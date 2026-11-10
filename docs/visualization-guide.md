# Visualization guide

## Purpose and boundaries

`src/visualization/` is a read-only reporting layer. It reads saved CSV/JSON artifacts, the saved Excel evaluation workbook, the configured dataset, and quantified findings in `docs/results_investigation_report.md`. It does not import model-training, preprocessing, prediction-generation, or evaluation modules. It does not read saved model bundles, retrain models, regenerate metrics, or write inside an experiment run directory.

The CLI writes figures to `outputs/figures/<category>/` by default. An output path inside the selected experiment directory is rejected. PNG files use 300 DPI; SVG is available for vector editing and publication workflows.

## Module layout

| Module | Responsibility |
|---|---|
| `config.py` | Project paths, labels, colors, and figure defaults. |
| `loaders.py` | Read-only CSV/JSON/Excel loading, schema checks, numeric checks, and experiment manifests. |
| `styling.py` | Shared Seaborn/Matplotlib theme and PNG/SVG export. |
| `performance.py` | Overall primary, pooled-MCC, and support-weighted comparisons from the evaluation workbook. |
| `project_analysis.py` | Primary per-project MCC, paired model comparison, and project distribution charts. |
| `robustness.py` | Paired Random Forest-minus-Dummy changes across saved project-version metrics. |
| `ablation.py` | Feature-ablation macro MCC, pooled MCC, F1, deltas, and project spread. |
| `dataset.py` | Class balance, project sizes, and missingness among the predictors listed in run metadata. |
| `investigation.py` | Quantified feature associations, cross-project overlap counts, and documented integrity counts. |
| `cli.py` | Command-line selection, skip reporting, output routing, and format choice. |

## Figures and source artifacts

The CLI checks the selected run directory and generates only figures supported by valid source artifacts.

| Category | Figures | Required source |
|---|---|---|
| `performance` | `model_performance_comparison`, `model_mcc_comparison`, `pooled_mcc_comparison`, `support_weighted_model_performance` | `evaluation_workbook.xlsx`, `Model Comparison` sheet |
| `projects` | `project_mcc_comparison`, `project_model_comparison`, `project_performance_distribution` | `evaluation_workbook.xlsx`, `Model Comparison` sheet |
| `robustness` | `robustness_version_metric_deltas` | `robustness/version_robustness.csv` |
| `ablation` | `feature_ablation_mcc`, `feature_ablation_delta`, `feature_ablation_pooled_mcc`, `feature_ablation_f1`, `feature_ablation_project_mcc_distribution` | `feature_ablation/metadata.json` and the project, version, and macro metrics CSVs |
| `dataset` | `class_distribution`, `project_size_distribution`, `feature_missingness` | `metadata.json` and the dataset CSV (by default `resources/ds.csv`) |
| `investigation` | `feature_associations`, `cross_project_overlap`, `dataset_integrity_counts` | Quantified tables and counts in `docs/results_investigation_report.md` |

The dataset target view follows the run manifest's `buggy = bugs > 0` definition. The CSV reader treats `-` as missing, matching the project's loader. Missingness figures include only predictor columns listed in the run manifest.

The investigation report contains aggregate overlap and duplication counts, but it does not contain saved predictions split into overlapping and unseen feature-vector groups. The overlap-versus-unseen performance chart is therefore skipped. Saved prediction rows also lack a stable source-row identifier, and their project/version/class keys can repeat. The overlap count chart is descriptive and is not evidence of leakage or invalid evaluation.

The saved robustness outputs contain Random Forest and Dummy metrics by project and project-version, but no alternative robustness configuration or repeated-run identifier. The available robustness chart therefore shows paired Random Forest-minus-Dummy changes by project-version and explicitly does not present them as configuration or repeated-run effects. A configuration comparison is reported as unavailable.

## Metric aggregation and interpretation

The figures preserve the aggregation used by each artifact:

- The evaluation workbook's primary metrics first average project-version scores equally within each project, then give every project equal weight in the macro average. The overall performance chart uses this primary aggregation. The per-project charts retain one project row per model.
- The workbook's support-weighted metrics weight versions by support within each project, then give every project equal weight. They appear in a separate supplementary chart.
- The workbook's `Pooled MCC` combines confusion counts within each project before computing MCC, then macro-averages informative projects. It appears separately from primary version-level MCC. Workbook rows with a non-`Available` status are treated as missing for charting, including zero placeholders for undefined MCC.
- Robustness diagnostic project metrics pool all saved observations within each held-out project; `robustness_summary.csv` then gives each project one contribution to summary statistics. Version-level robustness metrics retain missing versions as a separate group.
- Robustness version metrics contain one row for each project-version group, retaining missing versions. The robustness delta figure pairs saved Random Forest and Dummy values by project and version; these groups are not repeated experimental runs.
- Feature-ablation `MCC` first averages informative version-level MCC values within each project, then macro-averages across projects. `Pooled_MCC` is different: it combines confusion counts within each project before averaging project-level MCCs. These appear in separate figures.
- Feature-ablation F1 uses the version-level F1 values averaged within project, followed by an equal-project macro average.
- Undefined or non-informative MCC and specificity values remain missing. They are never replaced with zero to make a chart complete.

The robustness diagnostics and ablation outputs therefore have distinct metric aggregation methods. Figure titles and notes state the relevant aggregation. Do not compare their values as if they were calculated from the same aggregation.

## Commands

Run all available figures:

```bash
uv run python -m src.visualization.cli \
  --experiment-dir outputs/lopo_run_20261109_103355_224454
```

Generate one category:

```bash
uv run python -m src.visualization.cli \
  --experiment-dir outputs/lopo_run_20261109_103355_224454 \
  --category projects
```

Choose a separate output directory and export both formats:

```bash
uv run python -m src.visualization.cli \
  --experiment-dir outputs/lopo_run_20261109_103355_224454 \
  --output-dir outputs/figures \
  --format both
```

Optional source locations can be supplied explicitly:

```bash
uv run python -m src.visualization.cli \
  --experiment-dir outputs/lopo_run_20261109_103355_224454 \
  --dataset-path resources/ds.csv \
  --investigation-report docs/results_investigation_report.md
```

Options are `--category {all,performance,projects,robustness,ablation,dataset,investigation}`, `--format {png,svg,both}`, `--output-dir`, `--dataset-path`, and `--investigation-report`. The command lists generated paths, skipped figures, and reasons for skipped figures.

## Adding a visualization

1. Confirm the source artifact, schema, aggregation level, and missing-value semantics in the existing code and documentation.
2. Add a loader that validates the actual artifact columns and numeric ranges. Keep optional artifacts optional and return a clear skip reason when unavailable.
3. Put chart logic in the module that owns the chart category and use the shared style/export helper.
4. Keep aggregation in saved artifacts. If no result exists, do not calculate or invent one in the visualization layer.
5. Add tests with small temporary files and synthetic DataFrames. Check that source artifact bytes remain unchanged.
6. Update the CLI reporting, this guide, and the README only when the new visualization is part of the documented workflow.
