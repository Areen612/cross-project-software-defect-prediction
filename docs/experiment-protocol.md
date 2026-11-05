# Experiment Protocol

> **Project:** Cross-Project Software Defect Prediction  
> **Protocol status:** Planned; no experiment execution has been performed.  
> **Primary evaluation:** Leave-One-Project-Out (LOPO) across 39 projects.  
> **Primary model:** Random Forest.  
> **Baseline:** Majority-class Dummy Classifier.  
> **Primary comparison metric:** Matthews Correlation Coefficient (MCC).

## 1. Objective and scope

The experiment evaluates whether a defect prediction model trained on observations from multiple software projects can generalize to a project that is completely excluded from training.

The primary experiment is a Python implementation using a fixed set of approved predictors, fold-local preprocessing, and Leave-One-Project-Out evaluation.

The experiment is intended as a reproducible baseline within a 20-hour practice-project limit. It is not intended to exhaustively optimize model parameters or establish universal conclusions about cross-project defect prediction.

No model results should be reported until the complete protocol has been implemented, tested, and executed.

## 2. Dataset and target

### 2.1 Input dataset

- Input file: `resources/ds.csv`
- Observations: 110,917 rows
- Projects: 39
- Original columns: 29
- Each row is retained as a separate observation.
- Repeated class names across versions must not be deduplicated.
- The raw CSV must remain unchanged.

The first blank-header column may be read as `Unnamed: 0`. It is an undocumented index-like field and must not be used as a predictor.

### 2.2 Target construction

The source target column is `bugs`.

Construct the binary target as follows:

- `buggy = True` when `bugs > 0`
- `buggy = False` when `bugs == 0`

Before target construction:

- Reject missing target values.
- Reject non-numeric target values.
- Reject negative target values.
- Do not silently convert invalid target values into valid labels.

The `bugs` column must not be included in the predictor matrix.

### 2.3 Approved primary predictors

The primary experiment uses exactly these **18 predictors**:

```python
FEATURES = [
    "noc",
    "cbo",
    "rfc",
    "lcom",
    "ca",
    "ce",
    "npm",
    "lcom3",
    "loc",
    "dam",
    "moa",
    "mfa",
    "cam",
    "ic",
    "cbm",
    "amc",
    "nr",
    "ndc",
]
```

The feature list must be defined explicitly in configuration. Do not infer predictors by selecting all numeric columns.

### 2.4 Excluded variables and metadata

The following six consecutive metric columns are excluded from the primary predictor set following the supervisor's instruction, which specified the range from `nml` through `avg(cc)`:

```python
EXCLUDED_FEATURES = [
    "nml",
    "isOld",
    "ndpv",
    "ner",
    "max(cc)",
    "avg(cc)",
]
```

This is a supervisor-directed feature exclusion. It must not be described as proof that all six variables are almost entirely missing or empirically unimportant. The missingness rates differ substantially across these variables; only `avg(cc)` is extremely sparse in the inspected dataset.

The following columns are not predictors:

| Column | Role |
|---|---|
| `Project` | Outer-fold grouping and reporting |
| `Version` | Version-level evaluation grouping |
| `Class` | Observation identification and traceability |
| `bugs` | Source of the binary target |
| Blank-header first column | Undocumented index-like field; excluded |

The implementation must verify that all required metadata, target-source, and approved predictor columns exist before proceeding.

Unexpected columns should be reported but must never be silently added to the predictor set.

## 3. Data loading and validation

### 3.1 Loading

Load the CSV from the configured input path.

The loader must:

1. Check that the input file exists.
2. Reject a zero-byte input file.
3. Read the CSV into a pandas DataFrame.
4. Avoid unnecessary data transformations at the loading stage.

The loader should not construct the target, impute values, select predictors, or generate folds.

### 3.2 Dataset validation

Validation must check:

- The DataFrame contains observations.
- Column names are unique.
- All required columns exist.
- The target-source column contains no missing values.
- Target-source values are numeric and non-negative.
- Unexpected columns are reported.
- The approved predictor configuration contains exactly the 18 specified fields.

Missing predictor values are allowed at this stage. They are handled by fold-specific preprocessing.

### 3.3 Missing-value parsing

The dataset uses blank cells and `-` to represent missing metric values.

Both representations must be converted into proper missing values before numeric preprocessing.

The implementation must not convert missing predictor values to zero.

## 4. Outer evaluation: Leave-One-Project-Out

The primary evaluation uses all 39 projects as outer test folds.

For every distinct project:

1. Select the project as the held-out project.
2. Assign all its observations to the test set.
3. Assign all observations from the other 38 projects to the training set.
4. Fit preprocessing using training data only.
5. Fit the model using the transformed training data.
6. Transform and predict the held-out project.
7. Store predictions and fold metadata.
8. Continue until every project has been held out exactly once.

Expected outcome:

- 39 outer folds.
- Each project appears as the test project exactly once.
- Each test fold contains observations from only its held-out project.
- Training and test project sets are disjoint.

This is not a random row-level split and not a collection of source-project/target-project pair experiments.

Fold generation must be automated from the dataset's actual `Project` values. Project names must not be manually hard-coded.

## 5. Fold-local preprocessing

Preprocessing must be performed independently inside each outer fold.

For each fold:

1. Select the 18 approved predictors.
2. Convert blanks and `-` placeholders into missing values.
3. Inspect missingness in the training portion.
4. Identify predictors that are entirely missing in the training portion.
5. Drop those predictors for that fold and record their names.
6. Fit a median imputer using the remaining training predictors only.
7. Apply the fitted imputer to the held-out test project.
8. Preserve the feature ordering between training and test data.

No imputer or other learned preprocessing component may be fitted using the held-out project's observations.

The primary Random Forest pipeline does not require feature scaling. No scaler will be used for the primary model.

A feature that is entirely missing in a training fold must not be imputed using statistics from the held-out project.

The implementation must record fold-specific dropped features so that preprocessing decisions are auditable.

## 6. Primary model: Random Forest

Use `sklearn.ensemble.RandomForestClassifier`.

The initial configuration is fixed:

| Parameter | Value |
|---|---|
| `n_estimators` | `200` |
| `max_features` | `"sqrt"` |
| `class_weight` | `None` |
| `random_state` | `42` |
| `n_jobs` | `-1` |

These settings define the initial baseline experiment. They are not claimed to be optimal.

The model must be fitted separately in every outer fold using only the corresponding training projects.

The primary model must use the same 18-feature configuration in every fold, except where a feature is entirely missing in that fold's training data and must therefore be dropped according to the preprocessing policy.

## 7. Baseline model

Use:

```python
DummyClassifier(strategy="most_frequent")
```

The Dummy Classifier must be evaluated in every outer fold using the same training/test project separation.

It provides a majority-class reference against which the Random Forest results can be compared.

The baseline is important because the dataset is imbalanced:

- Buggy: 102,167 observations (92.11%)
- Non-buggy: 8,750 observations (7.89%)

Accuracy alone must not be used to conclude that the Random Forest is effective.

## 8. Optional sensitivity analysis

A class-weighted Random Forest may be considered after the primary Random Forest and Dummy baseline have been completed and reviewed.

If performed:

- It must be clearly labelled as a sensitivity analysis.
- It must use the same 39 outer folds.
- It must follow the same fold-local preprocessing rules.
- It must not replace or overwrite the primary Random Forest results.

Optional additional models must not delay the primary deliverables.

## 9. Prediction records

For each held-out observation, preserve enough information to trace the prediction back to its project and fold.

The prediction output should include:

- Held-out project
- Version
- Class identifier
- True target
- Predicted target
- Model name
- Fold/model identifier

The implementation should retain the original observation identity or a stable row index so predictions can be checked against the source records.

## 10. Evaluation metrics

Calculate the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)

MCC is the primary comparison metric because it accounts for all four confusion-matrix outcomes and is useful under class imbalance.

For each evaluation group, also retain:

- Total row count
- Buggy count
- Non-buggy count
- True negatives (TN)
- False positives (FP)
- False negatives (FN)
- True positives (TP)

Use a consistent zero-division convention for precision, recall, and F1. The planned convention is `zero_division=0`. Record this choice in the workbook and report.

If a version group contains only one actual target class, flag its MCC as **not informative for two-class discrimination**. Do not interpret a numerical MCC from a degenerate group as evidence of meaningful two-class discrimination.

## 11. Version-level evaluation

For every held-out project:

1. Group its predictions by `Version`.
2. Calculate all specified metrics separately for each version group.
3. Retain support and confusion-matrix counts.
4. Preserve missing version labels and represent them as `Version unavailable`.

Do not infer or fabricate missing version values.

The workbook's `Version Metrics` sheet must contain one row per held-out project-version group and model.

## 12. Project summaries

For each project, calculate the **unweighted arithmetic mean across its version groups** for each requested metric.

This means each version group contributes equally to the project summary, regardless of its observation count.

Label these values explicitly as `mean across versions`.

Do not confuse this calculation with computing a metric once over all observations belonging to a project.

The workbook's `Project Summary` sheet must retain the project-level summary and the aggregation convention.

## 13. Overall model comparison

The primary overall comparison is the **macro average across the 39 held-out project summaries**.

Each project receives equal weight, regardless of its size.

MCC is the primary comparison metric. Report accuracy, precision, recall, and F1 alongside it.

Compare:

- Primary Random Forest
- Majority-class Dummy baseline
- Any separately labelled sensitivity or optional models, if performed

### 13.1 Supplementary pooled evaluation

In addition to the primary macro project-level comparison, calculate supplementary pooled metrics by combining all held-out predictions from the 39 outer folds.

Calculate pooled metrics directly from the combined predictions and confusion-matrix counts.

Do not calculate pooled F1 or pooled MCC by taking a weighted or unweighted average of project-level F1 or MCC values.

Pooled accuracy is equivalent to the observation-weighted average of project accuracy, but pooled F1 and MCC generally are not equivalent to weighted averages of project-level scores.

Clearly label pooled results as supplementary and distinguish them from macro project-level results.

## 14. Output workbook

The required Excel workbook must contain the following sheets:

### `Version Metrics`

One row per held-out project-version group, including:

- Model
- Project
- Version
- Support
- Buggy count
- Non-buggy count
- Accuracy
- Precision
- Recall
- F1
- MCC
- TN
- FP
- FN
- TP

### `Project Summary`

One row per project and model, containing the unweighted arithmetic mean of version-level metrics. Clearly label the aggregation as `mean across versions`.

### `Model Comparison`

Include:

- Macro project-level model comparison (primary)
- Pooled observation-level model comparison (supplementary)
- Clear identification of the primary MCC metric
- Clear identification of the Random Forest and Dummy baseline

### `Run Notes`

Record:

- Dataset filename and checksum
- Dataset row and project counts
- Target definition
- The 18 approved predictors
- The six supervisor-directed exclusions
- Missing-value parsing rules
- Fold-local preprocessing decisions
- LOPO evaluation design
- Random Forest parameters
- Baseline configuration
- Metric definitions and zero-division convention
- Software and package versions
- Execution timestamp
- Relevant limitations and deviations

## 15. Model artifacts

Save each outer-fold Random Forest model together with the preprocessing components required to apply it.

Use a clear naming convention based on the held-out project.

Also train a separate final Random Forest using all eligible observations after the cross-project evaluation has been completed.

The all-data model is intended for future use. Its training performance must not be reported as cross-project evaluation performance.

Do not mix the final all-data model with the 39 outer-fold evaluation models.

## 16. Reproducibility and leakage prevention

The implementation must follow these rules:

1. Keep the raw CSV immutable.
2. Use the explicit approved 18-feature list.
3. Keep `Project`, `Version`, `Class`, `bugs`, and the blank-header column out of the predictors.
4. Do not reintroduce the six supervisor-excluded variables without a documented protocol change.
5. Generate outer folds automatically from the 39 projects.
6. Never allow observations from the held-out project into training or preprocessing.
7. Fit imputation using training projects only.
8. Do not use the held-out project's target or predictor distribution to make preprocessing decisions.
9. Use fixed and recorded model settings.
10. Preserve predictions and fold metadata.
11. Keep the final all-data model separate from outer-fold models.
12. Record deviations from this protocol, including whether they occurred before or after inspecting results.

## 17. Testing and quality assurance

Before running the full experiment, automated tests should verify:

- Dataset loading and missing-file handling.
- Duplicate-column detection.
- Required-column validation.
- Target validation and target construction.
- Exact 18-feature predictor configuration.
- Exclusion of the six supervisor-directed variables.
- Exclusion of metadata and target-source fields from predictors.
- Correct generation of 39 LOPO folds.
- Disjoint training and test project sets.
- Correct handling of missing predictor values.
- Fold-local imputation.
- Fold-specific removal of entirely missing training features.
- Confusion-matrix accounting.
- Version-level metric calculations.
- Project mean-across-versions calculations.
- Macro project aggregation.
- Pooled metric calculations.
- Workbook sheet presence and expected coverage.

The full experiment must not be run until the relevant tests and protocol checks pass.

## 18. Required deliverables and completion review

The primary experiment is complete only when the following deliverables are available:

- Reproducible experiment script.
- 39-fold LOPO evaluation.
- Random Forest predictions and metrics.
- Majority-class Dummy baseline predictions and metrics.
- Excel workbook with all required sheets.
- Saved fold-specific models and preprocessing.
- Separately labelled final all-data Random Forest model.
- Run metadata and configuration.
- Test results and final quality-assurance review.

The final review must confirm:

- Exactly 39 outer folds were executed.
- Every project was held out exactly once.
- No training/test project leakage occurred.
- Every expected project-version group is represented.
- The project summaries use the specified unweighted mean across versions.
- Macro project averages are calculated from project summaries.
- Pooled metrics are calculated from combined held-out predictions.
- Model artifacts and preprocessing settings are traceable.
- The raw dataset was not modified.

## 19. Limitations

- The original dataset provenance, license, metric definitions, and extraction procedure have not yet been established.
- Project-level differences may affect cross-project generalization.
- The target is based on recorded bug counts and does not establish defect severity or discovery timing.
- LOPO evaluates transfer to unseen projects in this dataset; it does not by itself demonstrate future-version prediction for an already known project.
- Version groups with very small support may produce unstable metrics.
- Median imputation is a practical baseline and does not explain the missing-data mechanism.
- Feature importance from Random Forest is not causal evidence.
- The supervisor-directed exclusion of six variables is a predefined feature-selection decision, not evidence that those variables have no predictive value.

---

**Protocol boundary:** This document defines the planned experiment and its implementation requirements. It contains no trained-model results or claims about achieved predictive performance.