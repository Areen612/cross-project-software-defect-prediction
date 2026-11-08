
from pathlib import Path

import pandas as pd


def build_predictions_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Combine LOPO fold results into one DataFrame.

    Each row represents one held-out class observation, retaining
    its project, version, class identifier, actual label, and prediction.

    Project and Version are preserved because evaluation will be
    performed separately for each project-version combination.
    """
    if not results:
        raise ValueError("Experiment results cannot be empty.")

    fold_predictions = []

    for result in results:
        metadata_test = result["metadata_test"]
        actual_series = result["y_test"].rename("Actual")

        predicted_series = pd.Series(
            result["predictions"],
            index=actual_series.index,
            name="Predicted",
        )

        fold_dataframe = pd.concat(
            [metadata_test, actual_series, predicted_series],
            axis=1,
        )

        fold_predictions.append(fold_dataframe)

    combined_dataframe = pd.concat(fold_predictions, axis=0)

    return combined_dataframe[
        ["Project", "Version", "Class", "Actual", "Predicted"]
    ].reset_index(drop=True)


def save_predictions(
    fold_results: list[dict],
    output_dir: str | Path,
    filename: str = "predictions.csv",
) -> Path:
    """
    Save LOPO experiment predictions to a CSV file.

    Parameters
    ----------
    fold_results : list[dict]
        Results from each LOPO fold.
    output_dir : str or Path
        Directory where the predictions CSV will be saved.
    filename : str, optional
        Output CSV filename.

    Returns
    -------
    Path
        Path to the saved predictions CSV file.
    """
    predictions_df = build_predictions_dataframe(fold_results)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    predictions_df.to_csv(output_path, index=False)

    return output_path