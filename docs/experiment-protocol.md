# Experiment Protocol

> **Project:** Cross-Project Software Defect Prediction  
> **Protocol status:** Planned experiment; no model training or experiment execution has been performed.  
> **Primary evaluation:** Leave-One-Project-Out (LOPO)  
> **Primary model:** RandomForestClassifier  
> **Baseline:** Majority-class DummyClassifier  
> **Time constraint:** 20-hour practice experiment  
> **Dataset description:** See `dataset-and-methodology.md`.

## 1. Experimental objective

The experiment evaluates whether a defect prediction model trained on multiple software projects can generalize to a project that was not represented during training.

The primary evaluation uses **Leave-One-Project-Out (LOPO)** across all 39 projects. Each project serves as the held-out test project exactly once. The remaining 38 projects are pooled to form that fold's training set.

This is not a random row split and is not a collection of independent source-project/target-project pair experiments.

The experiment is intended to establish a reproducible cross-project baseline, not to exhaustively optimize models or establish universal conclusions about software defect prediction.

## 2. Dataset and feature requirements

The input dataset is `resources/ds.csv`. Its descriptive characteristics, target definition, missingness, feature inventory, and limitations are documented in `dataset-and-methodology.md`.

### 2.1 Target

Construct the binary target from `bugs`:

- `buggy = True` when `bugs > 0`.
- `buggy = False` when `bugs == 0`.

The `bugs` column must not be included among predictors.

### 2.2 Primary predictors

Use exactly these 21 professor-approved predictors:

`noc`, `cbo`, `rfc`, `lcom`, `ca`, `ce`, `npm`, `lcom3`, `loc`, `dam`, `moa`, `mfa`, `cam`, `ic`, `cbm`, `amc`, `nr`, `ndc`, `ndpv`, `ner`, `max(cc)`.

Do not select predictors automatically from all numeric columns.

The following fields must be excluded:

- Blank-header first column (`Unnamed: 0` in some readers).
- `Project`.
- `Version`.
- `Class`.
- `bugs`.
- `nml`.
- `isOld`.
- `avg(cc)`.

The professor explicitly instructed that `nml`, `isOld`, and `avg(cc)` be removed. These exclusions apply to the primary experiment.

`Project` is used to construct the outer folds. `Version` is used for reporting and grouping. `Class` is retained for traceability. None of these fields is a predictor.

## 3. Experimental design: Leave-One-Project-Out

### 3.1 Outer evaluation

The primary experiment uses **Leave-One-Project-Out (LOPO)** evaluation across all 39 projects.

For each outer fold:

1. Select one project as the held-out test project.
2. Place every row from that project in the test set.
3. Pool every row from the other 38 projects into the training set.
4. Fit preprocessing and the model using training data only.
5. Transform and predict the held-out project's observations.
6. Save predictions and fold metadata for evaluation.
7. Repeat until every project has served as the held-out project once.

This produces 39 outer folds and 39 fitted outer-fold models per algorithm.

The fold construction must be automated from the project values present in the dataset. Projects or versions must not be manually selected.

A fold integrity check must confirm:

- The training and test project sets are disjoint.
- The test set contains exactly the designated held-out project.
- No observations from the held-out project are included in training.
- Every project appears as the test project exactly once.

### 3.2 Fold-local preprocessing

For each outer fold:

1. Parse blank cells and `-` as missing values.
2. Determine which features are entirely missing in that fold's training data.
3. Drop those features for that fold and record their names.
4. Fit a median imputer on the remaining training features only.
5. Apply the fitted imputer to the held-out project.
6. Fit the model using the preprocessed training data.
7. Transform and predict the held-out project using the already-fitted preprocessing and model.

No imputer, scaler, feature selector, or model may be fitted using held-out project observations.

A feature that is entirely missing in the full dataset is not expected based on the dataset audit. However, a feature may be entirely missing in the training portion of an individual fold. The implementation must handle and record that situation.

### 3.3 Scaling

Random Forest does not require feature scaling. No scaler is used in the primary Random Forest pipeline.

Any optional model that requires scaling must fit its scaler using the outer training projects only.

### 3.4 Hyperparameter tuning

The initial experiment uses a fixed, simple Random Forest configuration.

If hyperparameter tuning is later attempted:

- Validation must be grouped by project within the outer training projects.
- The held-out outer project must remain untouched until final fold evaluation.
- The search must remain small and computationally proportionate to the 20-hour project limit.
- Any tuning procedure and its settings must be documented.

Do not use the held-out project to select hyperparameters, preprocessing decisions, or thresholds.

## 4. Primary model: Random Forest

The primary model is `RandomForestClassifier`.

Random Forest is selected as a practical baseline for heterogeneous software metrics because it can model nonlinear decision boundaries and feature interactions without requiring standardized features or normally distributed inputs.

The initial configuration is:

| Parameter | Value |
|---|---|
| `n_estimators` | `200` |
| `max_features` | `"sqrt"` |
| `class_weight` | `None` |
| `random_state` | `42` |
| `n_jobs` | `-1` |

This configuration is a reproducible baseline, not a claim that the settings are optimal.

The complete model configuration and software versions must be recorded with the results.

## 5. Majority-class baseline

A `DummyClassifier` using a majority-class strategy will be evaluated in every outer fold.

It provides a reference for the performance obtainable by predicting the majority class without learning relationships between metrics and defects.

The baseline is important because the target is highly imbalanced. Random Forest performance should be interpreted in relation to this baseline, particularly for MCC, recall, precision, and F1.

The Dummy baseline must use the same outer folds and evaluation procedure as the primary model.

## 6. Class-weighted sensitivity analysis

The primary Random Forest uses `class_weight=None`.

If the baseline and primary results indicate that class imbalance materially limits minority-class detection, a class-weighted Random Forest may be evaluated as a separately labelled sensitivity analysis.

The sensitivity analysis must:

- Use the same outer LOPO folds.
- Use fold-local preprocessing.
- Be reported separately from the primary Random Forest.
- Preserve the primary results rather than replacing or overwriting them.

This analysis is optional and must not delay the primary deliverables.

## 7. Evaluation and aggregation

### 7.1 Project-version metrics

For every held-out project, predictions will be grouped by `Version`.

Missing version values will be retained and labelled `Version unavailable`. No version label will be inferred or fabricated.

For each project-version group, calculate:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Matthews Correlation Coefficient (MCC).
- Row count (support).
- Buggy count.
- Non-buggy count.
- Confusion-matrix counts: TN, FP, FN, TP.

The implementation must explicitly document the zero-division behavior used for precision, recall, and F1.

A consistent choice is `zero_division=0` for precision, recall, and F1. This convention must be stated in the workbook and report.

If a version group contains only one actual target class, flag its MCC as **not informative for two-class discrimination**. A numerical MCC returned by a library in a degenerate case must not be interpreted as evidence of meaningful two-class discrimination.

### 7.2 Project summaries

For each project, calculate the unweighted arithmetic mean of each requested metric across that project's version groups.

For example, if Project A has three versions, its project-level F1 is:

`Project A F1 = (F1_v1 + F1_v2 + F1_v3) / 3`

Label these values explicitly as **mean across versions**.

Every version group contributes equally to its project's summary, regardless of its row count. This follows the evaluation structure described by the professor.

This project-level average is distinct from calculating metrics once over all observations belonging to the project.

Both the version-level support counts and the aggregation rule must be retained so readers can understand this choice.

### 7.3 Macro project-level model comparison

For the primary overall comparison, calculate the arithmetic mean of the 39 project summaries, giving every project equal weight.

For example:

`Macro project F1 = (Project A F1 + Project B F1 + ... + Project AM F1) / 39`

Here, each project contributes equally, regardless of its number of observations.

**MCC is the primary comparison metric.** Accuracy, precision, recall, and F1 should be reported alongside it.

The comparison must distinguish:

- Primary Random Forest.
- Majority-class Dummy baseline.
- Class-weighted Random Forest, if performed.
- Optional secondary models, if performed.

This macro project-level result answers: how well does the model perform across projects when each project is treated as equally important?

### 7.4 Supplementary pooled evaluation

In addition to the macro project-level comparison, calculate supplementary pooled metrics using all held-out predictions from the 39 LOPO folds.

Each observation appears exactly once in the combined held-out predictions because every project serves as the test project exactly once.

For pooled evaluation:

1. Combine the held-out predictions from all 39 projects.
2. Construct one overall confusion matrix using all combined predictions.
3. Calculate accuracy, precision, recall, F1, and MCC from the combined counts.
4. Report the total number of observations and the overall confusion-matrix counts.

Pooled evaluation gives larger projects greater influence because they contribute more observations.

For accuracy, pooled evaluation is mathematically equivalent to weighting project-level accuracy by project size.

For F1 and MCC, pooled metrics must be calculated from the combined confusion-matrix counts. They must not be obtained by simply taking an observation-weighted average of the individual project F1 or MCC scores, because these metrics are nonlinear and the calculations are generally not equivalent.

Pooled results must be clearly labelled **pooled across all held-out observations**. They must not be confused with the macro average across projects.

The macro project-level results remain the primary overall comparison. Pooled results provide a complementary observation-level perspective.

## 8. Optional models

The following models are optional and must not delay the primary Random Forest, Dummy baseline, workbook, and required review.

### 8.1 HistGradientBoostingClassifier

Histogram-based gradient boosting is a tree-boosting alternative that can model nonlinear relationships and supports missing values natively in scikit-learn.

It may provide a useful comparison with Random Forest.

If evaluated:

- Use the exact same outer LOPO folds.
- Document its missing-value handling as a distinct preprocessing strategy.
- Do not describe its native missing-value handling as equivalent to median imputation.

### 8.2 LinearSVC

`LinearSVC` provides a linear-margin classification comparison.

It is sensitive to feature scales, so it requires:

- Fold-local imputation.
- Fold-local robust scaling.

Both preprocessing steps must be fitted only on the training projects in each outer fold.

`LinearSVC` does not provide native calibrated probabilities by default. The initial comparison can use its class predictions for the specified classification metrics. Any probability-based metrics or threshold-dependent decisions must be handled carefully and documented.

### 8.3 Model scope

No neural network is planned for this 20-hour practice experiment.

Optional models must not take priority over completing and reviewing the primary experiment.

## 9. Reproducibility and implementation requirements

The implementation should follow these principles:

1. **Keep raw data immutable.** Never overwrite `resources/ds.csv`. Derived data, logs, predictions, models, and workbooks belong in a separate output directory.
2. **Separate responsibilities.** Keep data loading and auditing, preprocessing, model construction, evaluation, and reporting in separate modules.
3. **Use explicit configuration.** Store the random seed, model parameters, target rule, feature exclusions, and metric conventions in a reproducible configuration.
4. **Avoid leakage.** Fit all learned preprocessing and model components using outer-training projects only.
5. **Automate folds.** Derive the 39 project folds from the dataset; do not hard-code a hand-picked list of projects.
6. **Test invariants.** Add tests for target construction, feature exclusion, project disjointness, fold counts, missing-value handling, confusion-matrix accounting, and aggregation.
7. **Record provenance.** Record input filename, file checksum, row count, project count, package versions, execution timestamp, and configuration with each run.
8. **Preserve artifacts.** Save fold models together with their preprocessing, clearly naming each artifact by held-out project. Save the final all-data model separately and label it clearly.
9. **Do not confuse fitting and evaluation.** The all-data model is for future use; its training scores are not cross-project evaluation results.
10. **Keep outputs traceable.** Predictions should include project, version, class identifier, true target, predicted target, and fold/model identifier, subject to appropriate data-handling rules.
11. **Use version control responsibly.** Keep virtual environments, secrets, raw datasets without redistribution permission, and generated model artifacts out of Git.
12. **Document deviations.** Any change to the protocol must be recorded with its rationale and whether it was decided before or after inspecting results.

## 10. Planned deliverables

The required experiment is expected to produce:

### 10.1 Experiment implementation

- A reproducible Python script that runs the complete 39-fold LOPO experiment.
- Tests covering the key data, fold, preprocessing, and evaluation invariants.

### 10.2 Excel workbook

An Excel workbook containing:

| Sheet | Required contents |
|---|---|
| `Version Metrics` | One row per held-out project-version group, with metrics and support/confusion counts. |
| `Project Summary` | Unweighted mean metrics across versions for each project. |
| `Model Comparison` | Macro project-level results as the primary comparison, plus supplementary pooled metrics calculated from all held-out predictions. |
| `Run Notes` | Target definition, exclusions, missing-value handling, split design, model settings, metric conventions, and limitations. |

### 10.3 Model artifacts

- Saved fold-specific Random Forest models and their preprocessing, named by held-out project.
- A separately labelled final Random Forest fitted on all eligible rows for future use.
- Run metadata and review checks confirming all 39 folds, project separation, workbook coverage, aggregation consistency, and saved settings.

The final all-data model is a deployment or future-use artifact. Its training scores must not be presented as cross-project evaluation results.

### 10.4 Output location

All generated artifacts should be written to a new output directory within the project workspace. The raw CSV must remain unchanged.

## 11. Required review before considering the experiment complete

After the primary experiment has run, review all of the following:

- [ ] All 39 projects were held out exactly once.
- [ ] Every fold has disjoint training and test project sets.
- [ ] Each test fold contains only its designated held-out project.
- [ ] No held-out project contributed to preprocessing or model fitting.
- [ ] Only the 21 approved predictors entered the primary model.
- [ ] `nml`, `isOld`, `avg(cc)`, identifiers, and `bugs` were excluded from predictors.
- [ ] Any feature dropped because it was entirely missing in a fold's training data was recorded.
- [ ] All expected project-version groups appear in the workbook.
- [ ] Missing version values are labelled consistently.
- [ ] Confusion-matrix counts reconcile with group support.
- [ ] Project summaries match the unweighted means across version groups.
- [ ] Model comparison uses the specified project-level macro aggregation.
- [ ] The primary Random Forest and Dummy baseline are both represented.
- [ ] Saved models include their corresponding preprocessing.
- [ ] The final all-data model is stored separately from fold models.
- [ ] Run settings, package versions, and input checksum are recorded.
- [ ] The original CSV remains unchanged.
- [ ] Supplementary pooled metrics are calculated from the combined held-out predictions, not from weighted averages of project F1 or MCC scores.

## 12. Experiment constraints and limitations

- The 20-hour practice experiment prioritizes a rigorous and reproducible baseline over exhaustive optimization.
- The dataset's original provenance, license, metric definitions, and collection procedures remain unverified. See `dataset-and-methodology.md`.
- Project-level differences may reflect language, architecture, project size, development practices, metric extraction tools, or data collection choices. Model performance may therefore be affected by dataset shift.
- A Random Forest's feature importance is not causal evidence and can be biased by correlated predictors and feature characteristics.
- LOPO measures transfer to unseen projects in this dataset. It does not by itself demonstrate future-version prediction for a known project.
- Version groups with very small support can yield unstable metrics. Support counts must accompany group-level scores.
- Imputation can conceal informative missingness. The primary median-imputation pipeline is a baseline, not a definitive solution to the missing-data mechanism.
- No results or performance claims should be written before the corresponding experiment has actually been executed and reviewed.

## 13. Protocol change management

Any change to the experimental design after this protocol is finalized must be documented.

For each change, record:

- What changed.
- Why the change was necessary.
- Whether it was decided before or after inspecting experimental results.
- Which results or artifacts are affected.

The primary protocol should remain distinguishable from sensitivity analyses and optional model comparisons.

---

**Protocol boundary:** This document defines the planned experimental procedure and required deliverables. It contains no trained-model results, tuned hyperparameters, or claims about achieved predictive performance.