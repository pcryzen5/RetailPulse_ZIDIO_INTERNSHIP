"""
Unit tests for churn modeling and inactivity risk.
"""
import pytest
import pandas as pd
import numpy as np

from src import config
from src.churn import compute_inactivity_risk_tiers, predict_churn_probability


def test_churn_artifacts_exist():
    """Verify that churn model and metrics json exist."""
    assert config.CHURN_MODEL_PATH.exists()
    assert config.CHURN_METRICS_PATH.exists()


def test_inactivity_risk_tiers():
    """Verify that inactivity tiers are correctly assigned without nulls."""
    df_rfm = pd.read_parquet(config.RFM_CUSTOMERS_PARQUET)
    df_tiers = compute_inactivity_risk_tiers(df_rfm)

    assert "Inactivity_Risk_Tier" in df_tiers.columns
    assert df_tiers["Inactivity_Risk_Tier"].isna().sum() == 0

    valid_tiers = [
        "Active / Very Recent",
        "Stable / Moderate Recency",
        "Inactivity Risk (90-180 days)",
        "High Inactivity (180-365 days)",
        "Dormant (> 1 year)"
    ]
    assert set(df_tiers["Inactivity_Risk_Tier"].unique()).issubset(set(valid_tiers))


def test_churn_prediction_output():
    """Verify that churn probabilities are bounded between 0 and 1."""
    synthetic_customers = pd.DataFrame({
        "Recency": [15, 250],
        "Frequency": [12, 1],
        "Monetary": [4500.0, 50.0],
        "Tenure_Days": [365, 300],
        "Avg_Order_Value": [375.0, 50.0],
        "Recent_30D_Spend": [800.0, 0.0],
        "Recent_30D_Tx": [2, 0]
    })
    probs = predict_churn_probability(synthetic_customers)
    assert len(probs) == 2
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    # Active high-value customer should have lower churn probability than inactive 1-time buyer
    assert probs[0] < probs[1]
