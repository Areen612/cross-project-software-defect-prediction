import pandas as pd

from scripts.run_feature_ablation import main


def test_cli_runs_feature_ablation(tmp_path, monkeypatch):
    from src.config import FEATURES

    rows = 6
    projects = ["A", "A", "B", "B", "C", "C"]
    metadata = pd.DataFrame({
        "Project": projects,
        "Version": ["1", "1", "2", "2", "3", "3"],
        "Class": ["A1", "A2", "B1", "B2", "C1", "C2"],
    })
    frame = metadata.copy()
    frame["bugs"] = [0, 1, 0, 1, 0, 1]
    for number, feature in enumerate(FEATURES):
        frame[feature] = [number + value for value in range(rows)]
    run_dir = tmp_path / "rf_run"
    run_dir.mkdir()
    reference = metadata.copy()
    reference["Actual"] = [False, True, False, True, False, True]
    reference["Predicted"] = reference["Actual"]
    reference.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "metadata.json").write_text("{}", encoding="utf-8")

    from src.evaluation import feature_ablation as module
    monkeypatch.setattr(module, "load_dataset", lambda: frame.copy())
    monkeypatch.setattr(module, "validate_dataset", lambda data: None)
    monkeypatch.setattr("sys.argv", ["run_feature_ablation", str(run_dir)])
    main()
    assert (run_dir / "feature_ablation" / "macro_metrics_all.csv").is_file()
