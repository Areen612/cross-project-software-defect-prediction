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

The purpose is to investigate whether model performance remains consistent across different held-out projects and versions, rather than relying exclusively on overall performance.

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

Feature ablation is a sensitivity analysis; it does not establish causal relationships between software metrics and defects.

## Evaluation metrics

The experiments use multiple classification metrics:

| Metric | Description |
|---|---|
| Accuracy | Proportion of correctly classified observations |
| Precision | Proportion of predicted buggy observations that are actually buggy |
| Recall | Proportion of actual buggy observations correctly identified |
| Specificity | Proportion of actual non-buggy observations correctly identified |
| Balanced Accuracy | Average of sensitivity and specificity |
| F1-score | Harmonic mean of precision and recall |
| MCC | Correlation between actual and predicted binary classifications |

Because the dataset is imbalanced, accuracy and F1-score are interpreted alongside specificity, balanced accuracy, and MCC.

MCC can be undefined for groups where the metric's mathematical denominator is zero. Undefined values are not treated as zero when calculating summary statistics.

Project-level macro averages give each project equal weight. They should not be confused with metrics calculated over all individual observations pooled together.

## Preliminary experimental results

The following results summarize the completed experiments.

### Random Forest robustness

Project-level averages:

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

### Interpretation

The preliminary results show strong Random Forest performance across the evaluated projects and versions, as well as limited average F1-score changes under the tested feature-removal configurations.

However, these findings are specific to the current dataset and evaluation protocol. They do not independently establish generalization to unrelated datasets or rule out possible dataset-related effects.

Further investigation is planned to examine cross-project feature overlap, duplicate observations, and dataset provenance.

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
- scikit-learn
- pytest
- uv
- Git and GitHub

## License

No license has currently been specified for this repository.