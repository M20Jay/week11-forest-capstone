# tests/test_train.py
# Sanity checks on the saved pipeline

import pytest
import joblib
import numpy as np
import pandas as pd
from src.forest.config import MODEL_PATH, FEATURES_PATH, FEATURE_COLUMNS

def test_pipeline_loads():
    pipeline = joblib.load(MODEL_PATH)
    assert pipeline is not None

def test_pipeline_predicts():
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    X = df[FEATURE_COLUMNS]
    predictions = pipeline.predict(X[:5])
    assert len(predictions) == 5

def test_predictions_are_binary():
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    X = df[FEATURE_COLUMNS]
    predictions = pipeline.predict(X)
    assert set(predictions).issubset({0, 1})

def test_probabilities_between_zero_and_one():
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    X = df[FEATURE_COLUMNS]
    probabilities = pipeline.predict_proba(X)[:, 1]
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0
