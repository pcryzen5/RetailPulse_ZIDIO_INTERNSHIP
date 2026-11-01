"""
Unit tests for customer segmentation and model artifacts.
"""
import pytest
import pandas as pd
import numpy as np

from src import config
from src.segmentation import load_segmentation_artifacts, predict_customer_segment


def test_segmentation_artifacts_loadable():
    """Verify that scaler, model, and metadata files exist and can be cleanly loaded."""
    assert config.RFM_SCALER_PATH.exists()
    assert config.KMEANS_MODEL_PATH.exists()
    assert config.SEGMENT_PROFILES_PATH.exists()

    kmeans, scaler, meta = load_segmentation_artifacts()
    assert kmeans.n_clusters == 4
    assert meta["n_clusters"] == 4
    assert "silhouette_score" in meta
    assert meta["silhouette_score"] > 0.35


def test_segment_assignment_consistency():
    """Verify that cluster labels in parquet match predictions from reloaded model."""
    kmeans, scaler, meta = load_segmentation_artifacts()
    df_rfm = pd.read_parquet(config.RFM_CUSTOMERS_PARQUET)

    # Test on sample
    sample = df_rfm.head(100)
    features = sample[config.CLUSTERING_FEATURES]
    scaled = scaler.transform(np.log1p(features))
    preds = kmeans.predict(scaled)

    assert np.array_equal(sample["Cluster"].values, preds)


def test_predict_customer_segment_function():
    """Verify that single customer inference returns expected dictionary structure."""
    result = predict_customer_segment(recency=10, frequency=15, monetary=3500.0)
    assert "cluster_id" in result
    assert "segment_name" in result
    assert result["cluster_id"] in [0, 1, 2, 3]
    assert len(result["scaled_features"]) == 3
