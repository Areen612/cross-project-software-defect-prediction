"""Shared paths, labels, and visual defaults for the read-only charts."""

from pathlib import Path

from src.config import FEATURES, INPUT_PATH, PROJECT_ROOT

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_DATASET_PATH = INPUT_PATH
DEFAULT_INVESTIGATION_REPORT = PROJECT_ROOT / "docs" / "results_investigation_report.md"

FIGURE_DPI = 300
DEFAULT_FORMAT = "png"
SUPPORTED_FORMATS = {"png", "svg"}

MODEL_LABELS = {
    "RandomForest": "Random Forest",
    "DummyClassifier": "Dummy baseline",
}
MODEL_COLORS = {
    "RandomForest": "#0072B2",
    "DummyClassifier": "#D55E00",
}
WORKBOOK_MODEL_KEYS = {
    "Random Forest": "RandomForest",
    "Dummy (most_frequent)": "DummyClassifier",
}
FEATURES_FOR_DATASET_CHARTS = tuple(FEATURES)


def category_output_dir(output_dir: str | Path, category: str) -> Path:
    """Return the category directory without creating or changing it."""
    return Path(output_dir) / category
