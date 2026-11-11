# Software Defect Prediction — Investigation Report

**Scope:** Completed read-only investigation findings and proposed investigations for next week.

> **Status note:** The findings below come from a completed read-only review of the dataset, evaluation workbook, saved predictions, methodology, and current code. The proposed next-week investigations have **not** been performed. No files, models, or experiment results were changed or rerun during the investigation.

## 1. Investigation summary

The investigation found no evidence of direct target leakage in the configured predictor list or reviewed target-construction path. However, several unresolved data-integrity and evaluation-validity questions remain, particularly around repeated observations, missing version identifiers, and exact feature-vector overlap across projects.

The LOPO implementation and fold-local preprocessing code follow the intended project-level separation design. The Random Forest substantially outperforms the majority-class Dummy baseline on the reported macro metrics, but its exceptionally high performance warrants further investigation before strong cross-project generalization claims are made.

## 2. Target definition and predictor selection

The target is defined as:

```python
buggy = bugs > 0
```

The experiment uses an explicit list of 18 predictors:

`noc`, `cbo`, `rfc`, `lcom`, `ca`, `ce`, `npm`, `lcom3`, `loc`, `dam`, `moa`, `mfa`, `cam`, `ic`, `cbm`, `amc`, `nr`, `ndc`.

`bugs`, `Project`, `Version`, and `Class` are not included in that predictor list. The reviewed code constructs the target from `bugs` and selects predictors using the configured feature list.

**Finding:** The reviewed configuration and data-preparation code provide no evidence of direct target leakage through inclusion of `bugs` or metadata as predictors. This does not rule out indirect leakage, dataset construction issues, or unusually strong relationships among selected metrics.

## 3. Predictor–target relationships

Directional ROC AUC is reported as `max(AUC, 1 − AUC)`. Spearman correlation was calculated on observed feature values. These are descriptive associations; they do not establish causation or leakage.

| Feature | Directional AUC | Spearman ρ |
|---|---:|---:|
| `loc` | 0.9474 | 0.4304 |
| `noc` | 0.9458 | 0.4236 |
| `ca` | 0.8886 | 0.3630 |
| `ce` | 0.8736 | 0.3557 |
| `lcom3` | 0.8611 | 0.3377 |
| `lcom` | 0.8366 | 0.3145 |
| `moa` | 0.8184 | 0.2973 |
| `nr` | 0.8058 | 0.3007 |
| `amc` | 0.8056 | 0.3035 |
| `dam` | 0.7647 | −0.2549 |
| `cbo` | 0.7618 | 0.2514 |
| `ndc` | 0.7437 | 0.2276 |
| `ic` | 0.7121 | 0.2019 |
| `npm` | 0.6225 | 0.1221 |
| `cbm` | 0.5919 | 0.0862 |
| `mfa` | 0.5648 | 0.0719 |
| `cam` | 0.5573 | 0.0870 |
| `rfc` | 0.5207 | 0.0450 |

Additional measurements:

- `loc`, `noc`, and `ca` had median within-project directional AUCs of 0.923, 0.961, and 0.927, respectively.
- No single feature achieved perfect threshold separation.
- The highest best-threshold balanced accuracy was 0.884, for `loc`.
- Missingness indicators for selected features had near-chance AUCs; the maximum was 0.5006.
- The six excluded metric columns had directional AUCs ranging from 0.514 to 0.548.

**Interpretation:** Several metrics have strong individual associations with the target, but these measurements alone do not establish leakage. They also do not explain by themselves why the combined Random Forest results are so high.

## 4. Dataset duplicates, repeated keys, and cross-project overlap

The investigation identified:

- **6,369 rows** have no `Version` value, from `velocity`, `xalan`, and `xerces`.
- **1,702 repeated `(Project, Version, Class)` keys**, accounting for **5,504 row occurrences**. All are among rows with missing versions.
- Eight repeated keys have both buggy and non-buggy labels: six in `xalan`, two in `xerces`.
- **772 rows** belong to exact duplicate groups after excluding the index-like column. These occur in `xalan` and `xerces`.
- Exact matches on all 18 selected feature values occur across projects: **4,104 feature-vector groups** span multiple projects and cover **30,521 rows**.
- Across all repeated feature-vector groups, none had conflicting target labels.
- Class names did not occur under more than one project.
- The blank-header index-like field has 53,657 distinct values ranging from 0 to 53,656, with 90,216 rows in repeated-value groups.
- The methodology document describes that field as ranging from 0 to 7,499, which does not match the current dataset.

**Interpretation and limitation:** Cross-project feature-vector matches may be legitimate, but they raise the question of how much held-out data has feature representations already present in training projects. Repeated keys and duplicate rows cannot be classified as accidental duplication versus legitimate snapshots without dataset provenance. No deduplication was performed.

## 5. Class balance

The dataset contains:

- **110,917 observations**
- **102,167 buggy observations (92.11%)**
- **8,750 non-buggy observations (7.89%)**
- **39 projects**, of which 37 contain both classes
- `ckjm` and `student-26` contain only buggy observations
- Of 93 project-version groups, 90 contain both classes and three contain only buggy observations

**Interpretation:** The strong class imbalance explains why a majority-class classifier can achieve high accuracy. Accuracy must therefore be interpreted alongside class-sensitive metrics and the confusion matrix.

## 6. Saved model results and Dummy baseline

In the latest reviewed run, `lopo_run_20261111_111341_636088`, both saved model outputs contain 110,917 predictions across 39 projects. Their observation counts match the source observations when compared as a multiset of `Project`, `Version`, `Class`, and actual label. The Dummy output predicts buggy for all 110,917 observations.

The workbook reports these macro averages, where version metrics are averaged within each project and projects receive equal weight:

| Metric | Random Forest | Dummy baseline |
|---|---:|---:|
| Accuracy | 99.7259% | 93.3493% |
| Precision | 99.7307% | 93.3493% |
| Recall | 99.9724% | 100.0000% |
| F1 | 99.8501% | 96.5045% |
| MCC | 0.9781 | No informative groups |

The saved Random Forest predictions have pooled confusion counts:

| Outcome | Count |
|---|---:|
| True negatives (TN) | 8,691 |
| False positives (FP) | 59 |
| False negatives (FN) | 9 |
| True positives (TP) | 102,158 |
| **Total errors** | **68** |

The submitted Random Forest and Dummy prediction files, workbook summaries, robustness summary, and feature-ablation summaries were reconciled against the reported results during the latest artifact review. This was an alignment check against the submitted artifacts, not a complete independent reproduction of the experiment.

**Interpretation and limitation:** Random Forest substantially outperforms the majority-class baseline on the reported macro metrics. However, the baseline comparison alone cannot establish that the model has learned robust cross-project patterns or rule out dataset-related effects.

## 7. LOPO and preprocessing review

The reviewed implementation:

- Uses `LeaveOneGroupOut` with `Project` as the grouping variable.
- Holds out each project as a test group.
- Fits the median imputer on training data only.
- Applies the fitted imputer to the held-out test data.
- Fits the model on training rows.
- Validates that every project is held out once and every observation is tested once.

**Finding:** The reviewed code path follows the intended project-level separation and fold-local preprocessing design. This was a code-path review; the folds were not rerun during the investigation.

## 8. Reporting and audit limitations

- Saved predictions contain no stable source-row identifier, only `Project`, `Version`, `Class`, `Actual`, and `Predicted`. Those keys can repeat, so saved output cannot uniquely identify every repeated source row.
- The alignment check compares identifiers and actual labels, not predictor values.
- `Version Metrics` contains Random Forest project-version results, and `Project Summary` contains Random Forest summaries derived from those version metrics.
- `Model Comparison` contains Random Forest and Dummy comparison results and macro averages. `Project Pooled` contains pooled metrics for both models by project and their separate equal-project macro averages.
- The workbook has no dataset-wide pooled-observation evaluation level, as specified by the protocol. Its project-pooled metrics and macro averages remain distinct from the version-derived project summaries.
- Run Notes records scikit-learn 1.9.1 but does not record the dataset checksum.
- `validate_dataset()` is defined, but the normal experiment entry point does not call it before target construction and feature selection.
- For this dataset, the selected features parsed as numeric and had no infinite values. There are 1,589 missing values across `rfc`, `lcom3`, `amc`, and `nr`.
- Dataset provenance and the exact observational meaning of rows remain unverified.

The dataset mirror and repository copy had the same SHA-256 hash (`90e885…3166f1`). The attached workbook and repository output workbook also matched (`e30d26…5119d39`).

## 9. Overall conclusions

The completed investigation supports these bounded conclusions:

1. No direct target leakage was identified in the explicit predictor list or reviewed target-construction path.
2. Several predictors have strong individual associations with defect labels, but no single predictor achieved perfect separation.
3. The dataset is highly imbalanced, materially affecting baseline accuracy.
4. Repeated keys, exact duplicate groups, missing version values, and extensive cross-project feature-vector overlap require further investigation. Their meaning cannot be determined from the available provenance.
5. The reviewed LOPO and fold-local imputation implementation follows the intended separation design.
6. Random Forest results are exceptionally high and outperform the Dummy baseline on the reported macro metrics, but further analyses are needed before making strong generalization claims.
7. Current prediction and workbook artifacts have auditability limitations.

These findings do **not** establish that the current results are invalid, nor do they establish that they are fully explained by genuine cross-project generalization.

## 10. Proposed investigation plan for next week

The following tasks are proposed for next week only. They have **not** been performed.

### Priority 1 — Cross-project feature-overlap analysis

- For each held-out project, calculate the proportion of test observations whose exact 18-feature vector appears in the training projects.
- Separate held-out observations into feature vectors also present in training and feature vectors not observed in training.
- Compare Random Forest performance, confusion counts, and class-sensitive metrics between those groups.
- Summarize overlap and performance by held-out project.
- Preserve the original experiment outputs and save this as a separate analysis.

**Research question:** How much of the reported performance is achieved on feature vectors already represented in other projects?

### Priority 2 — Dataset duplication and provenance investigation

- Inspect repeated `(Project, Version, Class)` keys, especially the eight keys with conflicting labels.
- Examine missing-version records from `velocity`, `xalan`, and `xerces`.
- Review exact duplicate groups and determine whether they could represent repeated snapshots or accidental duplication.
- Investigate the mismatch between the index-like field's documented and observed ranges.
- Seek authoritative dataset documentation or source provenance before deciding how repeated records should be treated.
- Do not remove or deduplicate observations without a documented rationale and a separately evaluated experiment.

**Research question:** What does each row represent, and which repeated observations are legitimate?

### Priority 3 — Follow-up robustness analysis, if indicated

- Review the robustness and feature-ablation results from the current experiment work before deciding what additional analyses are needed.
- Examine performance by project and by class, not only through macro averages.
- Investigate feature importance as a diagnostic, not as proof of causality.
- Consider whether performance remains strong when individual predictors or groups of highly associated predictors are removed.
- If additional experiments are justified, compare them against the unchanged Random Forest reference run and Dummy baseline, using separate output directories.

**Research question:** Do the current conclusions remain stable under reasonable alternative feature sets and evaluation views?

---

## 11. Reproducibility notes

- Preserve the original Random Forest and Dummy baseline runs as reference artifacts.
- Do not overwrite existing predictions or workbooks when running new investigations.
- Use separate output directories for any new overlap or robustness analyses.
- Record the dataset checksum, code revision, library versions, configuration, and run metadata for future experiments.
- Retain stable source-row identifiers in future prediction outputs where feasible, so repeated source observations can be audited individually.
- Document any dataset filtering or deduplication decision before applying it.
- Treat descriptive feature associations as evidence for further investigation, not as proof of leakage or causation.

**Current status:** Initial experiments, the evaluation workbook, Dummy baseline, and read-only investigation are complete. The next-week investigation priorities are cross-project feature overlap and dataset duplication/provenance, with follow-up robustness analysis guided by the results of the current experiment work.
