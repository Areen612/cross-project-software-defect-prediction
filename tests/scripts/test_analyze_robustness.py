import pandas as pd

from scripts.analyze_robustness import main


def _write_predictions(run_dir):
    (run_dir / "dummy_baseline").mkdir(parents=True)
    frame = pd.DataFrame({
        "Project": ["A", "A", "B", "B"],
        "Version": ["1", "1", "1", "1"],
        "Actual": [True, False, True, False],
        "Predicted": [True, False, True, True],
    })
    frame.to_csv(run_dir / "predictions.csv", index=False)
    frame.to_csv(run_dir / "dummy_baseline" / "predictions.csv", index=False)


def test_cli_generates_robustness_outputs(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_predictions(run_dir)
    monkeypatch.setattr("sys.argv", ["analyze_robustness", str(run_dir)])
    main()
    assert (run_dir / "robustness" / "robustness_summary.csv").is_file()


def test_cli_accepts_custom_output_directory(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_predictions(run_dir)
    output_dir = tmp_path / "diagnostics"
    monkeypatch.setattr(
        "sys.argv",
        ["analyze_robustness", str(run_dir), "--output-dir", str(output_dir)],
    )
    main()
    assert (output_dir / "project_robustness.csv").is_file()
