import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def processed_data():
    """Load a small sample of the census data and process it for training."""
    project_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path).sample(n=1000, random_state=42)
    X, y, _, _ = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    return X, y


def test_train_model_uses_random_forest(processed_data):
    """train_model should return a RandomForestClassifier."""
    X, y = processed_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_array_of_correct_length(processed_data):
    """inference should return a numpy array with one prediction per row."""
    X, y = processed_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y)


def test_compute_model_metrics_expected_values():
    """compute_model_metrics should return known values for a known example."""
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(0.6667, abs=1e-4)
