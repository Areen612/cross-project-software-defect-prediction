# Dataset and Methodology

> **Project:** Cross-Project Software Defect Prediction  
> **Document status:** Initial research plan; no model training or experiment execution has been performed.  
> **Dataset file:** `resources/ds.csv`  
> **Repository visibility:** Private  
> **Dataset provenance:** Not yet established; source details must be added before publication or formal research reporting.

## 1. Project purpose

This project investigates whether software defect prediction models trained on multiple software projects can generalize to a project that was not represented during training.

The central evaluation setting is **cross-project generalization**. Instead of randomly splitting class observations, the experiment holds out one complete project at a time. This prevents classes from the held-out project from entering the training set and better reflects the challenge of applying a model to a previously unseen project.

This document records the dataset characteristics and the planned methodology. It is a protocol, not a report of experimental results. No model fitting, hyperparameter tuning, or prediction is authorized by this document.

## 2. Dataset overview

The supplied CSV was inspected descriptively. The following facts were verified from the file:

| Property | Observed value |
|---|---:|
| Data rows (treated by this protocol as class records) | 110,917 |
| Distinct `Project` values | 39 |
| CSV columns | 29 |
| First four metadata-like columns | blank header, `Project`, `Version`, `Class` |
| Target-source column | `bugs` |
| Metric-named columns in the CSV | 24 |
| Approved primary predictors after supervisor-directed exclusions | 18 |
| Buggy observations (`bugs > 0`) | 102,167 (92.11%) |
| Non-buggy observations (`bugs == 0`) | 8,750 (7.89%) |
| Smallest project | 10 observations |
| Largest project | 23,058 observations |
| Mean project size | 2,844.03 observations |

Each CSV row contains values under `Project`, `Version`, and `Class` and is retained as a separate record. The source documentation has not been identified, so the precise observational unit and the meaning of these fields remain unverified. Do not deduplicate rows by `Class` alone.

The first header cell is blank; common CSV readers may display it as `Unnamed: 0`. Its values repeat and range from 0 to 7,499 in the inspected file, so its origin is not established and it is excluded from predictors. `Project`, `Version`, and `Class` are used only for grouping, reporting, and traceability; their exact source semantics remain unverified.

### 2.1 Dataset provenance and licensing

The dataset was supplied as a CSV, but its original publication, repository, collection procedure, license, and metric-generation tool have not yet been identified. The project must not attribute the data to a paper, benchmark, or repository until that provenance is verified.

Before using the dataset in a publication or making the repository public:

- identify the original dataset source and version, if possible;
- record the original URL, citation, access date, and license;
- verify whether redistribution is permitted;
- document how the CSV was obtained and whether it has been transformed;
- retain the supplied file unchanged as the raw input.

The repository is intended to remain **private** during initial development. The dataset should be excluded from Git history unless redistribution rights are confirmed. Making a repository private does not itself establish permission to redistribute third-party data.

## 3. Target definition

The binary target is derived from `bugs`:

- `buggy = True` when `bugs > 0`;
- `buggy = False` when `bugs == 0`.

The observed target distribution is:

| Target | Count | Percentage |
|---|---:|---:|
| Buggy | 102,167 | 92.11% |
| Non-buggy | 8,750 | 7.89% |
| Total | 110,917 | 100% |

The target is strongly imbalanced. Accuracy alone is therefore insufficient: a classifier that predicts the majority class can obtain high accuracy while failing to identify non-buggy classes. Evaluation will include precision, recall, F1, and Matthews Correlation Coefficient (MCC), alongside accuracy and confusion-matrix counts.

The original `bugs` count is used only to construct the target. It must not be included among predictors because doing so would directly leak the target information.

## 4. Dataset quality and distribution characteristics

### 4.1 Missing values

In the inspected CSV, missing metric values are represented by blank cells or `-`. Convert both to proper missing values during data loading. Missingness must be assessed after parsing, and no imputation statistic may be learned from a held-out project.

The following full-dataset missingness rates were measured:

| Feature | Missing observations | Missing percentage |
|---|---:|---:|
| `avg(cc)` | 104,880 | 94.56% |
| `ner` | 35,492 | 32.00% |
| `ndpv` | 34,690 | 31.28% |
| `nml` | 32,947 | 29.70% |
| `max(cc)` | 26,773 | 24.14% |
| `isOld` | 24,125 | 21.75% |
| `lcom3` | 1,162 | 1.05% |
| `rfc` | 303 | 0.27% |
| `amc` | 62 | 0.06% |
| `nr` | 62 | 0.06% |

The other metric columns have no missing values in the inspected CSV.

The six variables `nml`, `isOld`, `ndpv`, `ner`, `max(cc)`, and `avg(cc)` are excluded from the primary predictor set following the supervisor's instruction. The wording of the instruction identifies a range beginning with `nml` and ending with `avg(cc)`; the six consecutive metric columns in that range are therefore excluded.

This is a **supervisor-directed feature exclusion**, not a conclusion that all six variables are almost entirely missing or empirically unimportant. Their missingness rates differ substantially: `avg(cc)` is extremely sparse, while the other five have approximately 68–78% observed values. No feature-importance or causal analysis has been performed to establish that these variables are unimportant.

In the inspected CSV, `avg(cc)` is entirely missing in 38 of the 39 projects and has 104,880 missing values overall (94.56%). This is a verified dataset observation, separate from the supervisor's exclusion decision.

No feature is entirely empty in the full dataset. Nevertheless, a feature may be entirely missing in the training portion of an individual outer fold. Such a feature must be dropped for that fold and the feature name and held-out project must be recorded.

Missingness may itself reflect differences in data collection, metric extraction, or project characteristics. Median imputation is a practical baseline, but it does not explain why values are missing. The report should describe this limitation and avoid implying that imputation recovers the unknown true values.

### 4.2 Long-tailed values and extreme observations

Several metrics have upper tails substantially longer than their central ranges. The following descriptive statistics were measured from the supplied file. They are included to characterize scale and skew, not to define outlier cutoffs.

| Feature | Median | 99th percentile | Maximum |
|---|---:|---:|---:|
| `noc` | 3 | 42 | 413 |
| `ce` | 2 | 612 | 59,977 |
| `lcom` | 10 | 70 | 860 |
| `npm` | 1 | 48 | 860 |
| `moa` | 72 | 1,784.84 | 23,683 |
| `ndc` | 17 | 168.5 | 3,492 |
| `loc` | 2 | 34 | 347 |

These observations indicate substantial differences between typical and extreme values for some metrics. A large value is not, by itself, evidence of an invalid measurement. Large classes, highly coupled classes, and unusual project structures may legitimately produce extreme metric values.

**Primary handling policy:**

1. Do not remove observations solely because a metric value is large.
2. Do not apply global outlier thresholds using the complete dataset.
3. Inspect impossible values, parsing errors, unexpected infinities, and values that violate a metric's documented domain.
4. Record any exclusion or correction with the affected field, rule, number of observations, and justification.
5. Keep all defensible observations in the primary analysis.
6. If a transformation or outlier sensitivity analysis is later justified, report it separately from the primary result.

Random Forest does not require normally distributed features or feature standardization. It is therefore a reasonable first model for this dataset, although extreme values and data-quality errors still require inspection.

### 4.3 Project-size variation

Project sizes range from 10 to 23,058 observations, with a mean of approximately 2,844 observations. This variation has two implications:

- pooled row-level metrics can be dominated by large projects;
- a project-level macro average gives each held-out project equal weight in the overall comparison.

Both project-version results and project-level summaries should be reported so that the reader can inspect within-project variation without allowing the largest projects to dominate the headline comparison.

## 5. Feature inventory

The CSV contains 24 metric-named columns. Six consecutive variables, from `nml` through `avg(cc)` inclusive, are excluded following the supervisor's instruction. This leaves exactly **18 approved primary predictors**:

`noc`, `cbo`, `rfc`, `lcom`, `ca`, `ce`, `npm`, `lcom3`, `loc`, `dam`, `moa`, `mfa`, `cam`, `ic`, `cbm`, `amc`, `nr`, `ndc`.

The field names and their presence in the CSV are verified observations. The conventional readings below are tentative only: the original data dictionary, metric extractor, formulas, units, and aggregation rules have not been identified. These descriptions must not be cited as authoritative definitions for this dataset.

| Approved field | Conventional reading (unverified for this dataset) |
|---|---|
| `noc` | Number of children / immediate subclasses |
| `cbo` | Coupling between objects |
| `rfc` | Response for a class |
| `lcom` | Lack of cohesion in methods; exact variant unknown |
| `ca` | Afferent coupling |
| `ce` | Efferent coupling |
| `npm` | Number of public methods |
| `lcom3` | LCOM variant 3 |
| `loc` | Lines of code; counting convention unknown |
| `dam` | Data access metric; formula and scale unknown |
| `moa` | Measure of aggregation |
| `mfa` | Measure of functional abstraction |
| `cam` | Cohesion among methods |
| `ic` | Inheritance coupling |
| `cbm` | Coupling between methods |
| `amc` | Average method complexity |
| `nr` | Abbreviation not safely resolved |
| `ndc` | Abbreviation not safely resolved |

### 5.1 Predictor inclusion and exclusions

The primary predictor set is exactly the 18 approved fields listed above. Construct it explicitly; do not infer it by selecting every numeric column.

The following fields are excluded from predictors:

| Field | Reason |
|---|---|
| Blank-header first column (`Unnamed: 0` in some readers) | Meaning is undocumented; observed values repeat, so exclude as a predictor. |
| `Project` | Outer-fold grouping and reporting field; not a predictor. |
| `Version` | Reporting/grouping field; not a predictor. |
| `Class` | Record identifier/traceability field; not a metric predictor. |
| `bugs` | Used to construct the target; direct target leakage. |
| `nml` | Part of the supervisor-directed excluded range from `nml` through `avg(cc)`. |
| `isOld` | Part of the supervisor-directed excluded range from `nml` through `avg(cc)`. |
| `ndpv` | Part of the supervisor-directed excluded range from `nml` through `avg(cc)`. |
| `ner` | Part of the supervisor-directed excluded range from `nml` through `avg(cc)`. |
| `max(cc)` | Part of the supervisor-directed excluded range from `nml` through `avg(cc)`. |
| `avg(cc)` | Part of the supervisor-directed excluded range from `nml` through `avg(cc)`; also highly missing in the inspected CSV. |

The 18-field list above is the complete primary predictor set. This prevents excluded metrics, numeric identifiers, and the defect count from entering the model.

## 6. Experimental design: Leave-One-Project-Out

### 6.1 Outer evaluation

The primary experiment uses **Leave-One-Project-Out (LOPO)** evaluation across all 39 projects.

For each outer fold:

1. Select one project as the held-out test project.
2. Place every row from that project in the test set.
3. Pool every row from the other 38 projects into the training set.
4. Fit preprocessing and the model using training data only.
5. Transform and predict the held-out project's observations.
6. Save predictions and fold metadata for evaluation.
7. Repeat until every project has served as the held-out project once.

This produces 39 outer folds and 39 fitted outer-fold models per algorithm. It is **not** a random row split and it is **not** a collection of independent source-project/target-project pair experiments.

A fold integrity check must confirm that the training and test project sets are disjoint and that the test set contains exactly the held-out project. The process must be automated from the project values present in the dataset; projects or versions must not be manually selected.

### 6.2 Fold-local preprocessing

For each outer fold:

- Parse blanks and `-` as missing values.
- Determine features that are entirely missing in that fold's training data.
- Drop those features from that fold and record their names.
- Fit a median imputer on the remaining training features only.
- Apply the fitted imputer to the held-out project.
- Do not fit any imputer, scaler, selector, or model using held-out project observations.

Random Forest does not require scaling, so no scaler is used in the primary pipeline.

If hyperparameter tuning is later attempted, validation must be grouped by project within the outer training projects. The held-out outer project must remain untouched until final fold evaluation. Tuning should remain small and computationally proportionate to the 20-hour project limit.

## 7. Primary model: Random Forest

A `RandomForestClassifier` is the primary model.

Random Forest combines multiple decision trees trained on bootstrap samples and aggregates their predictions. It is a defensible baseline for heterogeneous software metrics because it can model nonlinear decision boundaries and feature interactions without requiring standardized features or normally distributed inputs.

The initial configuration is deliberately simple and reproducible:

| Parameter | Initial value |
|---|---|
| `n_estimators` | `200` |
| `max_features` | `"sqrt"` |
| `class_weight` | `None` |
| `random_state` | `42` |
| `n_jobs` | `-1` |

The full configuration and software versions must be recorded with the results. This is a baseline configuration, not a claim that these settings are optimal.

### 7.1 Majority-class baseline

A `DummyClassifier` using a majority-class strategy will be evaluated in every outer fold. It provides a simple reference for the performance obtainable by predicting the majority class without learning relationships between metrics and defects.

The baseline is essential because the target is highly imbalanced. A Random Forest result should be interpreted in relation to this baseline, particularly for MCC, recall, precision, and F1.

### 7.2 Class-weighted sensitivity analysis

The primary Random Forest run uses `class_weight=None`. If the baseline and primary results indicate that imbalance materially limits minority-class detection, a class-weighted Random Forest may be run as a separately labelled sensitivity analysis.

The sensitivity analysis must use the same outer project folds and fold-local preprocessing. It must not silently replace or overwrite the primary results.

## 8. Evaluation and aggregation

### 8.1 Project-version metrics

For every held-out project, predictions will be grouped by `Version`. Missing version values will be retained and labelled `Version unavailable`; no version label will be inferred or fabricated.

For each project-version group, calculate:

- Accuracy
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)
- Row count (support)
- Buggy count
- Non-buggy count
- Confusion-matrix counts: TN, FP, FN, TP

The implementation must explicitly document the zero-division behavior used for precision, recall, and F1. A consistent choice is `zero_division=0` for precision, recall, and F1, with the convention stated in the workbook and report.

If a version group contains only one actual target class, flag its MCC as **not informative for two-class discrimination**. The numerical MCC returned by a library in a degenerate case must not be interpreted as evidence of meaningful two-class discrimination.

### 8.2 Project summaries

For each project, calculate the unweighted arithmetic mean of each requested metric across that project's version groups. Label these values explicitly as **mean across versions**.

This gives every version group equal weight within a project, regardless of its row count. It is distinct from computing metrics once over all rows in a project. Both the version-level support counts and the aggregation rule must be retained so readers can understand this choice.

### 8.3 Model comparison

For overall model comparison, calculate macro averages across the 39 held-out projects so that large projects do not dominate the summary.

MCC is the primary comparison metric. Accuracy, precision, recall, and F1 should be reported alongside it. The comparison must distinguish the Random Forest from the majority-class Dummy baseline and, if performed, any sensitivity or optional model.

The aggregation level must be stated clearly. In particular, do not mix a macro average across projects with a pooled row-level metric and label both simply “average performance.”

## 9. Optional models (only after the primary deliverables)

The following models are optional and must not delay the Random Forest, Dummy baseline, workbook, and required review.

### 9.1 HistGradientBoostingClassifier

Histogram-based gradient boosting is a tree-boosting alternative that can model nonlinear relationships and supports missing values natively in scikit-learn. It may provide a useful comparison with Random Forest.

If evaluated, it must use the exact same outer LOPO folds. Its missing-value handling must be documented as a distinct preprocessing strategy rather than described as equivalent to median imputation.

### 9.2 LinearSVC

`LinearSVC` provides a linear-margin classification comparison. It is sensitive to feature scales, so it requires fold-local imputation and robust scaling. Both preprocessing steps must be fitted only on the training projects in each fold.

Any probability-based metrics or threshold-dependent decisions must be handled carefully because `LinearSVC` does not provide native calibrated probabilities by default. The initial comparison can use its class predictions for the specified classification metrics.

No neural network is planned for this 20-hour practice experiment.

## 10. Reproducibility and software-engineering practices

The implementation should follow these principles:

1. **Keep raw data immutable.** Never overwrite `resources/ds.csv`. Derived data, logs, predictions, models, and workbooks belong in a separate output directory.
2. **Separate responsibilities.** Keep data loading/auditing, preprocessing, model construction, evaluation, and reporting in separate modules.
3. **Use explicit configuration.** Store the random seed, model parameters, target rule, feature exclusions, and metric conventions in a reproducible configuration.
4. **Avoid leakage.** Fit all learned preprocessing and model components using outer-training projects only.
5. **Automate folds.** Derive the 39 project folds from the dataset; do not hard-code a hand-picked list of projects.
6. **Test invariants.** Add tests for target construction, feature exclusion, project disjointness, fold counts, missing-value handling, confusion-matrix accounting, and aggregation.
7. **Record provenance.** Record input filename, file checksum, row count, project count, package versions, execution timestamp, and configuration with each run.
8. **Preserve artifacts.** Save fold models together with their preprocessing, clearly naming each artifact by held-out project. Save the final all-data model separately and label it clearly.
9. **Do not confuse fit and evaluation.** The all-data model is for future use; its training scores are not cross-project evaluation results.
10. **Keep outputs traceable.** Predictions should include project, version, class identifier, true target, predicted target, and fold/model identifier, subject to appropriate data-handling rules.
11. **Use version control responsibly.** Keep virtual environments, secrets, raw datasets without redistribution permission, and generated model artifacts out of Git.
12. **Document deviations.** Any change to the protocol must be recorded with its rationale and whether it was decided before or after inspecting results.

## 11. Planned deliverables

The required experiment is expected to produce:

- A reproducible Python script that runs the complete 39-fold LOPO experiment.
- An Excel workbook containing:
  - `Version Metrics`: one row per held-out project-version group, with metrics and support/confusion counts.
  - `Project Summary`: unweighted mean metrics across versions for each project.
  - `Model Comparison`: macro project-level Random Forest and majority-baseline results.
  - `Run Notes`: target definition, exclusions, missing-value handling, split design, model settings, metric conventions, and limitations.
- Saved fold-specific Random Forest models and their preprocessing, named by held-out project.
- A separately labelled final Random Forest fitted on all eligible rows for future use.
- Run metadata and review checks confirming all 39 folds, project separation, workbook coverage, aggregation consistency, and saved settings.

All generated artifacts should be written to a new output directory within the project workspace. The raw CSV must remain unchanged.

## 12. Limitations and interpretation cautions

- The original dataset provenance and license are not yet known. Metric definitions and collection procedures must be verified before formal publication.
- Project-level differences may reflect language, architecture, project size, development practices, metric extraction tools, or data collection choices. A model's performance may therefore be affected by dataset shift.
- A random forest's feature importance is not causal evidence and can be biased by correlated predictors and feature characteristics.
- The target represents whether a class has a positive recorded bug count. It does not establish defect severity, discovery time, or whether the class was defect-prone before the observation period.
- LOPO measures transfer to unseen projects in this dataset. It does not by itself demonstrate future-version prediction for a known project.
- Version groups with very small support can yield unstable metrics. Support counts must accompany all group-level scores.
- Imputation can conceal informative missingness. The primary median-imputation pipeline is a baseline, not a definitive solution to the missing-data mechanism.
- A 20-hour practice experiment is intended to establish a rigorous, reproducible baseline, not to exhaustively optimize models or establish universal conclusions about defect prediction.

## 13. Open research questions / documentation tasks

Before finalizing the research report:

- [ ] Identify the dataset's original source, publication, repository, and license.
- [ ] Verify the definitions of the 18 included metrics before interpreting model effects, especially `nr` and `ndc`; confirm exact formulas for LCOM, LCOM3, DAM, MFA, CAM, IC, CBM, and AMC.
- [ ] Keep `nml`, `isOld`, `ndpv`, `ner`, `max(cc)`, and `avg(cc)` excluded unless a later, documented protocol change authorizes their reintroduction.
- [ ] Confirm the semantics and extraction process for the `Version` field.
- [ ] Document the dataset checksum and exact parsing rules in the run metadata.
- [ ] Confirm whether any metric values violate source-defined domains before deciding whether they are invalid.
- [ ] Finalize the protocol before running the experiment and record any subsequent deviations.

---

**Protocol boundary:** This document defines the dataset description and planned methodology only. It does not contain trained-model results, tuned hyperparameters, or claims about achieved predictive performance.