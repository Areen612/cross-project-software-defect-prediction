
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    RANDOM_SEED,
    N_ESTIMATORS,
    MAX_FEATURES,
    CLASS_WEIGHT,
    N_JOBS,
)


def create_random_forest() -> RandomForestClassifier:
    """
    Create a Random Forest classifier using the project's
    predefined experimental configuration.
    """
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features=MAX_FEATURES,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
    )

    return model
