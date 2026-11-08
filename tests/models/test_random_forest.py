
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    RANDOM_SEED,
    N_ESTIMATORS,
    MAX_FEATURES,
    CLASS_WEIGHT,
    N_JOBS,
)

from src.models.random_forest import create_random_forest


def test_returns_random_forest_classifier():
    model = create_random_forest()

    assert isinstance(model, RandomForestClassifier)


def test_uses_configured_hyperparameters():
    model = create_random_forest()

    assert model.n_estimators == N_ESTIMATORS
    assert model.max_features == MAX_FEATURES
    assert model.class_weight == CLASS_WEIGHT
    assert model.random_state == RANDOM_SEED
    assert model.n_jobs == N_JOBS


def test_model_is_not_fitted_when_created():
    model = create_random_forest()

    assert not hasattr(model, "estimators_")
