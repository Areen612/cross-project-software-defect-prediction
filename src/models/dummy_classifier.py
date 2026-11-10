"""Factory for the most-frequent-class Dummy baseline."""

from sklearn.dummy import DummyClassifier


def create_dummy_classifier() -> DummyClassifier:
    """
    Create a majority-class baseline.

    The majority class is learned from each fold's training labels only.
    No information from the held-out project is used to choose the class.
    """
    return DummyClassifier(strategy="most_frequent")
