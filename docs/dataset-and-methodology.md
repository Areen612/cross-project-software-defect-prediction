# Dataset and Methodology

> **Project:** Cross-Project Software Defect Prediction  
> **Document status:** Dataset audit and methodological background; no model training or experiment execution has been performed.  
> **Dataset file:** `resources/ds.csv`  
> **Repository visibility:** Private  
> **Dataset provenance:** Not yet established; source details must be added before publication or formal research reporting.

## 1. Project purpose

This project investigates whether software defect prediction models trained on multiple software projects can generalize to a project that was not represented during training.

The central evaluation setting is **cross-project generalization**. Instead of randomly splitting class observations, the planned experiment holds out one complete project at a time. This prevents classes from the held-out project from entering the training set and better reflects the challenge of applying a model to a previously unseen project.

This document records the dataset characteristics, target definition, feature inventory, and dataset-related methodological considerations. The actual experimental procedure is specified separately in `experiment-protocol.md`.

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
| Approved primary predictors after the professor's exclusions | 21 |
| Buggy observations (`bugs > 0`) | 102,167 (92.11%) |
| Non-buggy observations (`bugs == 0`) | 8,750 (7.89%) |
| Smallest project | 10 observations |
| Largest project | 23,058 observations |
| Mean project size | 2,844.03 observations |

Each CSV row contains values under `Project`, `Version`, and `Class` and is retained as a separate record. The source documentation has not been identified, so the precise observational unit and the meaning of these fields remain unverified. Do not deduplicate rows by `Class` alone.

The first header cell is blank; common CSV readers may display it as `Unnamed: 0`. Its values repeat and range from 0 to 7,499 in the inspected file, so its origin is not established and it is excluded from predictors.

`Project`, `Version`, and `Class` are used only for grouping, reporting, and traceability. Their exact source semantics remain unverified.

### 2.1 Dataset provenance and licensing

The dataset was supplied as a CSV, but its original publication, repository, collection procedure, license, and metric-generation tool have not yet been identified. The project must not attribute the data to a paper, benchmark, or repository until that provenance is verified.

Before using the dataset in a publication or making the repository public:

- Identify the original dataset source and version, if possible.
- Record the original URL, citation, access date, and license.
- Verify whether redistribution is permitted.
- Document how the CSV was obtained and whether it has been transformed.
- Retain the supplied file unchanged as the raw input.

The repository is intended to remain **private** during initial development. The dataset should be excluded from Git history unless redistribution rights are confirmed. Making a repository private does not itself establish permission to redistribute third-party data.

## 3. Target definition

The binary target is derived from `bugs`:

- `buggy = True` when `bugs > 0`.
- `buggy = False` when `bugs == 0`.

The observed target distribution is:

| Target | Count | Percentage |
|---|---:|---:|
| Buggy | 102,167 | 92.11% |
| Non-buggy | 8,750 | 7.89% |
| Total | 110,917 | 100% |

The target is strongly imbalanced. Accuracy alone is therefore insufficient: a classifier that predicts the majority class can obtain high accuracy while failing to identify non-buggy classes.

Evaluation will include precision, recall, F1, and Matthews Correlation Coefficient (MCC), alongside accuracy and confusion-matrix counts. The detailed evaluation and aggregation rules are defined in `experiment-protocol.md`.

The original `bugs` count is used only to construct the target. It must not be included among predictors because doing so would directly leak target information.

## 4. Dataset quality and distribution characteristics

### 4.1 Missing values

In the inspected CSV, missing metric values are represented by blank cells or `-`. Convert both to proper missing values during data loading.

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

The professor's reply explicitly excludes `nml`, `isOld`, and `avg(cc)`. In the inspected CSV, `avg(cc)` is entirely missing in 38 of the 39 projects and has 104,880 missing values overall (94.56%). This is a verified dataset observation, separate from the professor's feature-exclusion decision.

`nml` and `isOld` also have missing values, as shown above, but their missingness is not the reason for their exclusion; the professor explicitly instructed that they be removed.

No feature is entirely empty in the full dataset. Nevertheless, a feature may be entirely missing in the training portion of an individual outer fold. The handling of this situation is specified in `experiment-protocol.md`.

Missingness may itself reflect differences in data collection, metric extraction, or project characteristics. Median imputation is a practical baseline, but it does not explain why values are missing. The report should describe this limitation and avoid implying that imputation recovers the unknown true values.

### 4.2 Long-tailed values and extreme observations

Several metrics have upper tails substantially longer than their central ranges. The following descriptive statistics were measured from the supplied file. They characterize scale and skew; they do not define outlier cutoffs.

| Feature | Median | 99th percentile | Maximum |
|---|---:|---:|---:|
| `noc` | 3 | 42 | 413 |
| `ce` | 2 | 612 | 59,977 |
| `lcom` | 10 | 70 | 860 |
| `npm` | 1 | 48 | 860 |
| `moa` | 72 | 1,784.84 | 23,683 |
| `ndpv` | 0 | 653 | 16,890 |
| `ndc` | 17 | 168.5 | 3,492 |
| `loc` | 2 | 34 | 347 |

These observations indicate substantial differences between typical and extreme values for some metrics. A large value is not, by itself, evidence of an invalid measurement. Large classes, highly coupled classes, and unusual project structures may legitimately produce extreme metric values.

**Primary data-handling policy:**

1. Do not remove observations solely because a metric value is large.
2. Do not apply global outlier thresholds using the complete dataset.
3. Inspect impossible values, parsing errors, unexpected infinities, and values that violate a metric's documented domain.
4. Record any exclusion or correction with the affected field, rule, number of observations, and justification.
5. Keep all defensible observations in the primary analysis.
6. If a transformation or outlier sensitivity analysis is later justified, report it separately from the primary result.

Random Forest does not require normally distributed features or feature standardization. It is therefore a reasonable first model for this dataset, although extreme values and data-quality errors still require inspection.

### 4.3 Project-size variation

Project sizes range from 10 to 23,058 observations, with a mean of approximately 2,844 observations.

This variation has two implications:

- Pooled row-level metrics can be dominated by large projects.
- A project-level macro average gives each held-out project equal weight in the overall comparison.

Both project-version results and project-level summaries should be reported so that within-project variation can be inspected without allowing the largest projects to dominate the headline comparison.

## 5. Feature inventory

The CSV contains 24 metric-named columns. The professor's reply explicitly excludes `nml`, `isOld`, and `avg(cc)`, leaving exactly these **21 approved primary predictors**:

`noc`, `cbo`, `rfc`, `lcom`, `ca`, `ce`, `npm`, `lcom3`, `loc`, `dam`, `moa`, `mfa`, `cam`, `ic`, `cbm`, `amc`, `nr`, `ndc`, `ndpv`, `ner`, `max(cc)`.

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
| `ndpv` | Abbreviation not safely resolved |
| `ner` | Abbreviation not safely resolved |
| `max(cc)` | Possibly maximum cyclomatic complexity; meaning and aggregation level unknown |

### 5.1 Predictor inclusion and exclusions

The primary predictor set is exactly the 21 approved fields listed above. Construct it explicitly; do not infer it by selecting every numeric column.

The following fields are excluded from predictors:

| Field | Reason |
|---|---|
| Blank-header first column (`Unnamed: 0` in some readers) | Meaning is undocumented; observed values repeat, so exclude as a predictor. |
| `Project` | Outer-fold grouping and reporting field; not a predictor. |
| `Version` | Reporting/grouping field; not a predictor. |
| `Class` | Record identifier/traceability field; not a metric predictor. |
| `bugs` | Used to construct the target; direct target leakage. |
| `nml` | Explicitly excluded in the professor's reply. |
| `isOld` | Explicitly excluded in the professor's reply. |
| `avg(cc)` | Explicitly excluded in the professor's reply; also highly missing in the inspected CSV. |

The 21-field list above is the complete primary predictor set. This prevents excluded metrics, numeric identifiers, and the defect count from entering the model.

## 6. Dataset limitations and interpretation cautions

- The original dataset provenance and license are not yet known. Metric definitions and collection procedures must be verified before formal publication.
- Project-level differences may reflect language, architecture, project size, development practices, metric extraction tools, or data collection choices.
- The target represents whether a class has a positive recorded bug count. It does not establish defect severity, discovery time, or whether the class was defect-prone before the observation period.
- The exact semantics of `Project`, `Version`, and `Class` remain unverified.
- Imputation can conceal informative missingness. Median imputation is a baseline strategy, not a definitive solution to the missing-data mechanism.
- Large metric values should not automatically be interpreted as errors or removed without evidence.
- Conventional metric names and descriptions are not substitutes for the original dataset's data dictionary.

## 7. Open dataset documentation tasks

Before finalizing a research report:

- [ ] Identify the dataset's original source, publication, repository, and license.
- [ ] Verify the definitions of the 21 included metrics before interpreting model effects, especially `ndpv`, `ner`, `nr`, and `ndc`.
- [ ] Confirm exact formulas for LCOM, LCOM3, DAM, MFA, CAM, IC, CBM, AMC, and `max(cc)`.
- [ ] Keep `nml`, `isOld`, and `avg(cc)` excluded unless a later, documented protocol change authorizes their reintroduction.
- [ ] Confirm the semantics and extraction process for the `Version` field.
- [ ] Document the dataset checksum and exact parsing rules in run metadata.
- [ ] Confirm whether any metric values violate source-defined domains before deciding whether they are invalid.
- [ ] Retain the supplied raw CSV unchanged.

---

**Document boundary:** This document describes the dataset, target, predictors, and dataset-related limitations. The experimental procedure, model configuration, evaluation, and reporting requirements are defined in `experiment-protocol.md`.