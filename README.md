# Cross-Project Software Defect Prediction

## Overview

This project investigates **Cross-Project Software Defect Prediction (CPDP)** using software product metrics and a Random Forest classifier.

The objective is to evaluate whether a model trained on multiple software projects can predict defective classes in a previously unseen project.

The experiments use **Leave-One-Project-Out (LOPO) cross-validation**, where one project is held out for testing while the remaining projects are used for training. This process is repeated until every project has been used as the test project.

The project also investigates model performance through a Dummy baseline, robustness analysis, and feature ablation.

## Dataset

The dataset contains software metrics collected from multiple software projects and versions.

| Property | Value |
|---|---:|
| Observations | 110,917 |
| Projects | 39 |
| Project versions | 75 |
| Predictors | 18 |
| Target | `buggy` |

Each observation represents a software class.

### Target variable

The binary target variable `buggy` is derived from the `bugs` column:

- `buggy = 1` when `bugs > 0`
- `buggy = 0` when `bugs = 0`

The dataset is imbalanced, with approximately 92% buggy observations and 8% non-buggy observations.

### Selected predictors

The experiments use the following 18 software metrics:

```text
noc, cbo, rfc, lcom, ca, ce, npm, lcom3,
loc, dam, moa, mfa, cam, ic, cbm, amc, nr, ndc
```

The following columns are not used as predictors:

- `Project`, `Version`, and `Class`: identification and grouping information.
- `bugs`: used to construct the target variable.
- `nml`, `isOld`, `ndpv`, `ner`, `max(cc)`, and `avg(cc)`: excluded based on supervisor guidance.

The dataset also contains an index-like column that is not used for modeling.

## Methodology

### 1. Data preprocessing

Missing values are handled using median imputation.

Importantly, imputation is performed independently within each LOPO fold:

1. The median values are calculated using the training data only.
2. The fitted imputer transforms the training data.
3. The same fitted imputer transforms the held-out test project.

This prevents information from the test project from influencing the imputation process.

Features that contain no observed values in a training fold are dropped from that fold's modeling data.

### 2. Leave-One-Project-Out evaluation

The evaluation uses 39 LOPO folds.

For each fold:

1. One project is selected as the test project.
2. The other 38 projects form the training set.
3. Preprocessing is fitted using the training data.
4. A Random Forest classifier is trained.
5. Predictions are generated for the held-out project.

The process is repeated until all 39 projects have been held out once.

This evaluation design investigates cross-project generalization rather than randomly splitting observations from the same projects between training and testing.

### 3. Random Forest configuration

The primary classifier uses the following configuration:

| Parameter | Value |
|---|---|
| Model | Random Forest |
| Number of estimators | 200 |
| Maximum features | `sqrt` |
| Class weight | `None` |
| Random state | 42 |
| Number of jobs | -1 |

### 4. Dummy baseline

A Dummy classifier using the `most_frequent` strategy is evaluated using the same LOPO setup.

It provides a simple reference point by always predicting the majority class.

Comparing the Random Forest against this baseline helps determine whether the model learns useful distinctions beyond the dataset's majority-class distribution.

## Evaluation metrics and aggregation

The experiments use multiple classification metrics:

| Metric | What it evaluates |
|---|---|
| Accuracy | Overall proportion of correctly classified observations |
| Precision | How reliable the model's positive (buggy) predictions are |
| Recall | How many actual buggy observations the model identifies |
| Specificity | How many actual non-buggy observations the model identifies |
| Balanced Accuracy | Average performance across the buggy and non-buggy classes |
| F1-score | Balance between precision and recall |
| MCC | Overall correlation between actual and predicted classifications, accounting for all four confusion-matrix outcomes |

Because the dataset is imbalanced, accuracy and F1-score are interpreted alongside specificity, balanced accuracy, and MCC.

### Evaluation levels

The project reports two complementary aggregation paths from the same saved out-of-project predictions.

#### 1. Project-version level — detailed evaluation

Predictions are grouped by `Project` and `Version`. Metrics, support, class counts, and confusion-matrix counts are calculated for each group. Missing version labels are retained as a separate group.

#### 2. Project Summary — mean across versions

For each project, Accuracy, Precision, Recall, and F1 are calculated for each version and then averaged equally across that project's versions. MCC is averaged across informative version groups only. This is a mean of version-level scores, not a metric calculated over all project observations pooled together. Support-weighted version means are retained as supplementary fields.

#### 3. Project Pooled — supplementary evaluation

For each project, predictions from all of its versions are combined first. Accuracy, Precision, Recall, F1, MCC, and confusion-matrix counts are then calculated once from those pooled predictions. This is distinct from the Project Summary mean across versions.

### Macro-averages across projects

Both project-level paths have their own macro average:

- **Project-version-derived macro average:** average the project summaries from `Project Summary`, giving each project equal weight.
- **Project-pooled macro average:** average the metrics in `Project Pooled`, giving each project equal weight.

The workbook labels these paths separately. MCC averages exclude non-informative or undefined groups; such groups are not treated as ordinary zero-valued MCC results. Dataset-wide pooled metrics across all 39 held-out projects are out of scope and are not reported.

### Workbook aggregation details

| Workbook sheet | Calculation |
|---|---|
| `Version Metrics` | Metrics calculated separately for each project-version group. |
| `Project Summary` | Unweighted mean of version metrics within each project, plus supplementary support-weighted version means. |
| `Project Pooled` | Metrics calculated from raw predictions pooled across versions within each project, plus an equal-project macro average. |
| `Model Comparison` | Primary project-version-derived macro comparison. |
| `Run Notes` | Metadata and explicit descriptions of both aggregation paths. |

The workbook and robustness analysis distinguish the two paths explicitly. The robustness analysis's project metrics pool predictions across versions within each project, and its mean across project metrics is a macro-average of project-pooled metrics.

## Additional analyses

### Robustness analysis

The robustness analysis evaluates the saved Random Forest and Dummy baseline predictions without retraining the models.

It examines performance at two levels:

- **Project level:** Predictions are pooled across versions within each project.
- **Version level:** Performance is evaluated separately for each project version.

The analysis reports:

- Accuracy
- Precision
- Recall
- Specificity
- Balanced Accuracy
- F1-score
- Matthews Correlation Coefficient (MCC)

It also summarizes performance across projects and versions using descriptive statistics.

#### Interpretation

The Random Forest consistently outperformed the Dummy baseline across the evaluated projects and versions:

- At the project level, it achieved higher accuracy and F1-score in 37 of 39 projects and tied in the remaining two.
- At the version level, it performed better in 90 of 93 groups and tied in the other three.
- Its average specificity was 96.42%, compared with 0% for the Dummy baseline.
- Its average balanced accuracy was 98.19%, compared with 50% for the Dummy baseline.

The Dummy baseline predicted every observation as buggy. Although this gave it 100% buggy-class recall, it failed to identify any non-buggy observations.

The Random Forest's higher specificity and balanced accuracy indicate that it learned to distinguish between the two classes rather than simply favoring the majority class.

There was some variation between projects. The lowest Random Forest MCC was 0.805 for `student-22`, which contained only 37 observations, including three non-buggy observations. The model misclassified one of those three.

These results suggest strong and relatively consistent performance on the evaluated dataset. However, they do not establish generalization to unrelated datasets or rule out possible dataset-related effects.

### Feature ablation

Feature ablation investigates how sensitive model performance is to removing selected predictors.

Five configurations are evaluated using the same Random Forest settings and LOPO procedure:

| Configuration | Removed features |
|---|---|
| `baseline_all_features` | None |
| `without_loc` | LOC |
| `without_noc` | NOC |
| `without_loc_noc` | LOC, NOC |
| `without_top_association_features` | LOC, NOC, CA, CE |

Each configuration is trained and evaluated independently across all 39 folds.

The purpose is to examine whether the model's predictive performance depends heavily on these selected features.

#### Interpretation

Removing the selected predictors resulted in only small changes in average F1-score:

- Removing LOC reduced macro F1 by 0.035 percentage points.
- Removing NOC reduced macro F1 by 0.041 percentage points.
- Removing LOC and NOC together reduced it by 0.051 percentage points.
- Removing all four selected features (LOC, NOC, CA, CE) produced the largest reduction of 0.065 percentage points.

The combined removal of all four features changed 60 predictions out of 110,917. The effects varied across project versions: F1 decreased in 24 versions, remained unchanged in 64, and improved in 5.

These findings suggest that the model's average F1-score is relatively insensitive to the removal of the selected predictors under the current experimental setup.

Feature ablation is a sensitivity analysis; it does not establish causal relationships between software metrics and defects.

## Preliminary experimental results

The following results summarize the completed experiments.

### Random Forest robustness

The following figures are **macro-averages of project-level pooled metrics**. Predictions are pooled across versions within each project before metrics are calculated. Each project contributes equally to the macro-average.

| Metric | Random Forest | Dummy baseline |
|---|---:|---:|
| Accuracy | 99.72% | 93.30% |
| F1-score | 99.85% | 96.48% |
| Buggy-class recall | 99.96% | 100.00% |
| Non-buggy-class specificity | 96.42% | 0.00% |
| Balanced Accuracy | 98.19% | 50.00% |
| MCC | 0.978 | Undefined |

The Random Forest achieved higher accuracy and F1-score in 37 of 39 projects and tied in the remaining two. It also performed better in 90 of 93 project-version groups, tying in the other three.

The Dummy baseline predicted every observation as buggy. Consequently, it achieved perfect recall for the buggy class but failed to identify non-buggy observations.

### Feature ablation

| Configuration | Macro F1 | Change vs. baseline | Macro MCC |
|---|---:|---:|---:|
| Baseline | 99.850% | — | 0.9781 |
| Without LOC | 99.815% | -0.035 pp | 0.9776 |
| Without NOC | 99.809% | -0.041 pp | 0.9702 |
| Without LOC and NOC | 99.799% | -0.051 pp | 0.9747 |
| Without LOC, NOC, CA, CE | 99.785% | -0.065 pp | 0.9711 |

Removing all four selected predictors changed 60 predictions out of 110,917. The effects varied across project versions.

These findings suggest that the model's average F1-score is relatively insensitive to the removal of the selected predictors under the current experimental setup.

### Overall interpretation

The preliminary results show strong Random Forest performance across the evaluated projects and versions, as well as limited average F1-score changes under the tested feature-removal configurations.

The robustness analysis shows that the model outperforms a majority-class baseline across the evaluated projects and versions. The feature-ablation analysis suggests that its performance does not depend heavily on the four selected predictors.

However, these findings are specific to the current dataset and evaluation protocol. They do not independently establish generalization to unrelated datasets or rule out possible dataset-related effects.

## Project structure

```text
cross-project-software-defect-prediction/
│
├── docs/
│   ├── dataset-and-methodology.md
│   ├── experiment-protocol.md
│   └── results_investigation_report.md
│
├── notebooks/
│
├── outputs/
│
├── resources/
│   └── ds.csv
│
├── scripts/
│   ├── run_experiment.py
│   ├── run_dummy_baseline.py
│   ├── analyze_robustness.py
│   ├── run_feature_ablation.py
│   └── generate_workbook.py
│
├── src/
│   ├── config.py
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── preprocessing/
│   └── visualization/
│
├── tests/
│
├── pyproject.toml
└── README.md
```

## Installation

### Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)

### Set up the project

Clone the repository:

```bash
git clone https://github.com/Areen612/cross-project-software-defect-prediction.git

cd cross-project-software-defect-prediction
```

Install the project dependencies:

```bash
uv sync
```

Ensure the dataset is available at:

```text
resources/ds.csv
```

## Running the experiments

Commands should be executed from the project root.

### 1. Run the Random Forest LOPO experiment

```bash
uv run python -m scripts.run_experiment
```

### 2. Run the Dummy baseline

Provide the output directory of the corresponding Random Forest experiment:

```bash
uv run python -m scripts.run_dummy_baseline \
  outputs/lopo_run_20261109_103355_224454
```

### 3. Run robustness analysis

```bash
uv run python -m scripts.analyze_robustness \
  outputs/lopo_run_20261109_103355_224454
```

### 4. Run feature ablation

```bash
uv run python -m scripts.run_feature_ablation \
  outputs/lopo_run_20261109_103355_224454
```

### 5. Generate the evaluation workbook

```bash
uv run python -m scripts.generate_workbook \
  outputs/lopo_run_20261109_103355_224454
```

The commands above use the initial experiment output directory. Replace it with the appropriate experiment directory when running a new experiment.

## Outputs

Experiment outputs are stored under the `outputs/` directory.

The main experiment produces fold-level predictions and evaluation results.

The Dummy baseline produces its corresponding predictions and metrics.

The evaluation workbook contains:

- `Version Metrics`
- `Project Summary`
- `Model Comparison`
- `Run Notes`

The robustness analysis generates:

- `project_robustness.csv`
- `version_robustness.csv`
- `robustness_summary.csv`
- `README.txt`

The feature-ablation analysis generates:

- `project_metrics_all.csv`
- `version_metrics_all.csv`
- `macro_metrics_all.csv`
- `metadata.json`

It also stores separate result directories for each ablation configuration, including predictions, metrics, and configuration metadata.

## Visualizations

The read-only visualization module creates figures from the saved evaluation workbook, experiment CSV/JSON files, the configured dataset, and quantified findings in the investigation report. It does not train models, rerun evaluation, or change source artifacts. Missing or invalid inputs are reported and skipped.

Generate all available figures:

```bash
uv run python -m src.visualization.cli \
  --experiment-dir outputs/lopo_run_20261109_103355_224454
```

Generate one category, choose another figure directory, or export PNG and SVG:

```bash
uv run python -m src.visualization.cli \
  --experiment-dir outputs/lopo_run_20261109_103355_224454 \
  --category ablation \
  --output-dir outputs/figures \
  --format both
```

Categories are `performance`, `projects`, `robustness`, `ablation`, `dataset`, and `investigation`. Figures are saved under `outputs/figures/<category>/` by default. The command prints generated paths and explains skipped figures.

## Testing

The project uses pytest for automated testing.

Run the complete test suite:

```bash
uv run python -m pytest -v
```

Tests cover data loading and validation, preprocessing, model training, LOPO evaluation, metrics, prediction alignment, baseline evaluation, robustness analysis, feature ablation, and output generation.

## Current limitations and future investigation

Although the preliminary results are promising, several aspects require further investigation before drawing broader conclusions.

Planned investigation areas include:

1. **Cross-project feature overlap:** Examine similarities and repeated feature patterns across different projects.
2. **Dataset provenance:** Investigate the original dataset source and documentation.
3. **Duplicate observations:** Examine repeated records and their possible influence on evaluation results.
4. **Additional robustness checks:** Determine whether further experiments are needed based on the investigation findings.

The goal is to better understand the dataset and evaluate the reliability of the observed cross-project performance.

## Technologies

- Python
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- pytest
- uv
- Git and GitHub

## License

No license has currently been specified for this repository.