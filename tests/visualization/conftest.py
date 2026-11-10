from __future__ import annotations

import json

import pandas as pd
import pytest


@pytest.fixture
def robustness_run(tmp_path):
    run = tmp_path / "run"
    robustness = run / "robustness"
    robustness.mkdir(parents=True)
    project_rows = []
    version_rows = []
    for project, version, support, buggy, non_buggy in [
        ("A", "1", 4, 3, 1),
        ("B", None, 2, 2, 0),
    ]:
        for model, accuracy, precision, recall, specificity, balanced, f1, mcc in [
            ("RandomForest", 0.75, 0.75, 1.0, 0.0 if non_buggy else None, None, 0.857, 0.0 if non_buggy else None),
            ("DummyClassifier", buggy / support, buggy / support, 1.0, 0.0 if non_buggy else None, 0.5 if non_buggy else None, 0.8, None),
        ]:
            if non_buggy:
                balanced = ((recall or 0) + (specificity or 0)) / 2
            project_rows.append({
                "Project": project,
                "Model": model,
                "Support": support,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "Specificity": specificity,
                "Balanced_Accuracy": balanced,
                "F1": f1,
                "MCC": mcc,
                "Actual_Class_Status": "Both classes" if non_buggy else "Single class",
            })
            version_rows.append({
                "Project": project,
                "Version": version,
                "Model": model,
                "Support": support,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "Specificity": specificity,
                "Balanced_Accuracy": balanced,
                "F1": f1,
                "MCC": mcc,
                "Actual_Class_Status": "Both classes" if non_buggy else "Single class",
            })
    pd.DataFrame(project_rows).to_csv(robustness / "project_robustness.csv", index=False)
    pd.DataFrame(version_rows).to_csv(robustness / "version_robustness.csv", index=False)
    summary_rows = []
    for model in ("RandomForest", "DummyClassifier"):
        for metric, value in {
            "Accuracy": 0.75,
            "Precision": 0.75,
            "Recall": 1.0,
            "Specificity": 0.0,
            "Balanced_Accuracy": 0.5,
            "F1": 0.857,
            "MCC": 0.0 if model == "RandomForest" else None,
        }.items():
            summary_rows.append({
                "Model": model,
                "Metric": metric,
                "Projects_Total": 2,
                "Projects_With_Both_Classes": 1,
                "Projects_Contributing": 1 if metric in {"Specificity", "Balanced_Accuracy", "MCC"} else 2,
                "Mean": value,
                "Median": value,
                "Std": 0.0 if value is not None else None,
                "Min": value,
                "Q1": value,
                "Q3": value,
                "Max": value,
            })
    pd.DataFrame(summary_rows).to_csv(robustness / "robustness_summary.csv", index=False)
    (run / "metadata.json").write_text(json.dumps({
        "experiment": {"number_of_observations": 6},
        "target": {"column": "buggy", "definition": "buggy = bugs > 0"},
        "features": {"predictors": ["noc", "loc"]},
    }), encoding="utf-8")
    comparison_rows = []
    project_values = {
        "Random Forest": {
            "A": {"Accuracy": 0.75, "Precision": 0.75, "Recall": 1.0, "F1": 0.85,
                  "MCC": 0.5, "Pooled MCC": 0.4},
            "B": {"Accuracy": 1.0, "Precision": 1.0, "Recall": 1.0, "F1": 1.0,
                  "MCC": None, "Pooled MCC": None},
        },
        "Dummy (most_frequent)": {
            "A": {"Accuracy": 0.75, "Precision": 0.75, "Recall": 1.0, "F1": 0.8,
                  "MCC": None, "Pooled MCC": None},
            "B": {"Accuracy": 1.0, "Precision": 1.0, "Recall": 1.0, "F1": 1.0,
                  "MCC": None, "Pooled MCC": None},
        },
    }
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "MCC", "Pooled MCC"]
    for model, projects in project_values.items():
        for project, metrics in projects.items():
            for metric in metric_names:
                value = metrics[metric]
                comparison_rows.append({
                    "Model": model,
                    "Evaluation_Level": "Project",
                    "Project": project,
                    "Metric": metric,
                    "Value": value,
                    "Status": "Available" if value is not None else "No informative groups",
                    "Notes": "synthetic test fixture",
                })
            for metric in ("Accuracy", "Precision", "Recall", "F1"):
                value = metrics[metric]
                comparison_rows.append({
                    "Model": model,
                    "Evaluation_Level": "Project",
                    "Project": project,
                    "Metric": f"Support-weighted {metric}",
                    "Value": value,
                    "Status": "Available",
                    "Notes": "synthetic test fixture",
                })
        for metric in metric_names + [f"Support-weighted {name}" for name in ("Accuracy", "Precision", "Recall", "F1")]:
            values = [projects[name].get(metric, projects[name].get(metric.removeprefix("Support-weighted "))) for name in projects]
            valid = [value for value in values if value is not None]
            value = sum(valid) / len(valid) if valid else None
            comparison_rows.append({
                "Model": model,
                "Evaluation_Level": "Macro average",
                "Project": None,
                "Metric": metric,
                "Value": value,
                "Status": "Available" if value is not None else "No informative groups",
                "Notes": "synthetic test fixture",
            })
    pd.DataFrame(comparison_rows).to_excel(
        run / "evaluation_workbook.xlsx", sheet_name="Model Comparison", index=False
    )
    return run


@pytest.fixture
def ablation_run(robustness_run):
    run = robustness_run
    root = run / "feature_ablation"
    root.mkdir()
    configs = {
        "baseline_all_features": {"removed_features": []},
        "without_loc": {"removed_features": ["loc"]},
    }
    (root / "metadata.json").write_text(json.dumps({"configurations": configs}), encoding="utf-8")
    projects = []
    versions = []
    macros = []
    for name, adjustment in (("baseline_all_features", 0.0), ("without_loc", -0.01)):
        for project, score in (("A", 0.9 + adjustment), ("B", 0.8 + adjustment)):
            projects.append({
                "Configuration": name,
                "Project": project,
                "Support": 3,
                "F1": score,
                "MCC": score - 0.1,
                "Pooled_MCC": score - 0.11,
            })
            versions.append({
                "Configuration": name,
                "Project": project,
                "Version": "1",
                "Support": 3,
                "F1": score,
                "MCC": score - 0.1,
            })
        macros.append({
            "Configuration": name,
            "Project_Count": 2,
            "Accuracy": 0.85 + adjustment,
            "Precision": 0.9 + adjustment,
            "Recall": 0.95 + adjustment,
            "F1": 0.85 + adjustment,
            "MCC": 0.75 + adjustment,
            "Pooled_MCC": 0.74 + adjustment,
            "MCC_Informative_Projects": 2,
            "Pooled_MCC_Informative_Projects": 2,
        })
    pd.DataFrame(projects).to_csv(root / "project_metrics_all.csv", index=False)
    pd.DataFrame(versions).to_csv(root / "version_metrics_all.csv", index=False)
    pd.DataFrame(macros).to_csv(root / "macro_metrics_all.csv", index=False)
    return run


@pytest.fixture
def dataset_csv(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text(
        "Project,bugs,noc,loc\nA,1,2,10\nA,0,3,-\nB,2,4,30\n",
        encoding="utf-8",
    )
    return path
