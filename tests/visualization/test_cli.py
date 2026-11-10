from __future__ import annotations

from src.visualization.cli import main


def test_cli_generates_requested_category_and_reports_output(
    robustness_run, tmp_path, capsys
):
    source = robustness_run / "robustness" / "project_robustness.csv"
    original = source.read_bytes()
    output = tmp_path / "figures"
    result = main([
        "--experiment-dir", str(robustness_run),
        "--output-dir", str(output),
        "--category", "performance",
    ])
    captured = capsys.readouterr().out
    assert result == 0
    assert "Generated figures:" in captured
    assert (output / "performance" / "model_performance_comparison.png").is_file()
    assert source.read_bytes() == original


def test_cli_rejects_figure_output_inside_experiment(robustness_run, capsys):
    result = main([
        "--experiment-dir", str(robustness_run),
        "--output-dir", str(robustness_run / "figures"),
        "--category", "performance",
    ])
    assert result == 2
    assert "outside the experiment directory" in capsys.readouterr().out


def test_cli_reports_missing_optional_artifacts(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    result = main([
        "--experiment-dir", str(run),
        "--output-dir", str(tmp_path / "figures"),
        "--category", "robustness",
    ])
    assert result == 1
    assert "missing source file" in capsys.readouterr().out
