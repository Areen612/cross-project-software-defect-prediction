"""Command-line entry point for generating charts from saved artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from src.visualization.ablation import generate_ablation_figures
from src.visualization.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_INVESTIGATION_REPORT,
    DEFAULT_OUTPUT_DIR,
)
from src.visualization.dataset import generate_dataset_figures
from src.visualization.investigation import generate_investigation_figures
from src.visualization.loaders import (
    AblationArtifacts,
    ArtifactError,
    EvaluationArtifacts,
    RobustnessArtifacts,
    load_ablation_artifacts,
    load_evaluation_artifacts,
    load_robustness_artifacts,
    load_run_metadata,
)
from src.visualization.performance import generate_performance_figures
from src.visualization.project_analysis import generate_project_figures
from src.visualization.robustness import generate_robustness_figures


_CATEGORIES = ("performance", "projects", "robustness", "ablation", "dataset", "investigation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate read-only figures from saved software-defect prediction "
            "artifacts. This command does not train models or rerun evaluation."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Existing LOPO run directory containing saved metrics and metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Separate figure directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--category",
        choices=("all", *_CATEGORIES),
        default="all",
        help="Generate all available figures or one chart category.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=f"Optional source CSV path (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--investigation-report",
        type=Path,
        default=DEFAULT_INVESTIGATION_REPORT,
        help="Optional quantified investigation report in Markdown.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "svg", "both"),
        default="png",
        help="Output format; PNG is 300 DPI and SVG is vector output.",
    )
    return parser


def _formats(value: str) -> tuple[str, ...]:
    return ("png", "svg") if value == "both" else (value,)


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _run_category(
    name: str,
    operation: Callable[[], list[Path]],
    generated: list[Path],
    skipped: list[str],
) -> None:
    try:
        paths = operation()
    except (ArtifactError, ValueError, OSError) as exc:
        skipped.append(f"{name}: {exc}")
        return
    generated.extend(paths)
    if not paths:
        skipped.append(f"{name}: no figures were available from the source artifacts")


def _print_report(
    generated: list[Path],
    skipped: list[str],
    output_dir: Path,
) -> None:
    if generated:
        print("Generated figures:")
        for path in generated:
            print(f"  {path}")
    if skipped:
        print("Skipped:")
        for reason in skipped:
            print(f"  {reason}")
    print(f"Figure directory: {output_dir}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_dir = args.experiment_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if _inside(output_dir, experiment_dir):
        print("Error: figure output must be outside the experiment directory.")
        return 2
    if not experiment_dir.is_dir():
        print(f"Error: experiment directory not found: {experiment_dir}")
        return 2

    requested = _CATEGORIES if args.category == "all" else (args.category,)
    formats = _formats(args.format)
    generated: list[Path] = []
    skipped: list[str] = []
    evaluation: EvaluationArtifacts | None = None
    robustness: RobustnessArtifacts | None = None
    ablation: AblationArtifacts | None = None

    if set(requested) & {"performance", "projects"}:
        try:
            evaluation = load_evaluation_artifacts(experiment_dir)
        except ArtifactError as exc:
            for category in ("performance", "projects"):
                if category in requested:
                    skipped.append(f"{category}: {exc}")

    if evaluation is not None:
        if "performance" in requested:
            _run_category(
                "performance",
                lambda: generate_performance_figures(
                    experiment_dir, output_dir, formats, evaluation
                ),
                generated,
                skipped,
            )
        if "projects" in requested:
            _run_category(
                "projects",
                lambda: generate_project_figures(
                    experiment_dir, output_dir, formats, evaluation
                ),
                generated,
                skipped,
            )

    if "robustness" in requested:
        try:
            robustness = load_robustness_artifacts(experiment_dir)
        except ArtifactError as exc:
            skipped.append(f"robustness: {exc}")
        else:
            _run_category(
                "robustness",
                lambda: generate_robustness_figures(
                    experiment_dir, output_dir, formats, robustness
                ),
                generated,
                skipped,
            )
            skipped.append(
                "robustness configurations: unavailable; the saved robustness artifacts contain paired Random Forest and Dummy results only, with no configuration dimension or repeated runs."
            )


    if "ablation" in requested:
        try:
            ablation = load_ablation_artifacts(experiment_dir)
        except ArtifactError as exc:
            skipped.append(f"ablation: {exc}")
        else:
            _run_category(
                "ablation",
                lambda: generate_ablation_figures(
                    experiment_dir, output_dir, formats, ablation
                ),
                generated,
                skipped,
            )

    if "dataset" in requested:
        _run_category(
            "dataset",
            lambda: generate_dataset_figures(
                experiment_dir,
                output_dir,
                args.dataset_path,
                formats,
            ),
            generated,
            skipped,
        )

    if "investigation" in requested:
        try:
            metadata = load_run_metadata(experiment_dir)
            total_observations = int(metadata["experiment"]["number_of_observations"])
            if total_observations <= 0:
                raise ArtifactError("run metadata has a non-positive observation count")
            paths, notes = generate_investigation_figures(
                args.investigation_report,
                total_observations,
                output_dir,
                formats,
            )
            generated.extend(paths)
            skipped.extend(f"investigation: {note}" for note in notes)
        except (ArtifactError, KeyError, TypeError, ValueError, OSError) as exc:
            skipped.append(f"investigation: {exc}")

    _print_report(generated, skipped, output_dir)
    return 0 if generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
