"""
Unit tests for data and concept drift detection (PSI & KS-test).
"""
import pytest
import numpy as np
import pandas as pd

from src.drift import (
    calculate_psi,
    calculate_ks_test,
    evaluate_feature_drift,
    generate_drift_report
)


def test_psi_identical_distributions():
    """Identical distributions should exhibit near-zero PSI (< 0.05)."""
    np.random.seed(42)
    data = np.random.normal(loc=100, scale=15, size=5000)
    psi = calculate_psi(data, data)
    assert psi < 0.05, f"Expected near-zero PSI, got {psi}"


def test_psi_shifted_distribution():
    """Significantly shifted distribution should trigger high PSI (>= 0.25)."""
    np.random.seed(42)
    baseline = np.random.normal(loc=50, scale=10, size=5000)
    shifted = np.random.normal(loc=90, scale=20, size=5000)
    psi = calculate_psi(baseline, shifted)
    assert psi >= 0.25, f"Expected high PSI (>=0.25), got {psi}"


def test_ks_test_detection():
    """KS test should report no drift for identical distributions and detect shift for different distributions."""
    np.random.seed(42)
    dist1 = np.random.exponential(scale=2.0, size=2000)
    dist2 = np.random.exponential(scale=2.0, size=2000)
    dist3 = np.random.exponential(scale=6.0, size=2000)

    res_stable = calculate_ks_test(dist1, dist2)
    assert res_stable["p_value"] > 0.01

    res_drift = calculate_ks_test(dist1, dist3)
    assert res_drift["drift_detected"] is True
    assert res_drift["p_value"] < 0.001


def test_generate_drift_report_dataframe():
    """Verify that multi-feature drift report returns structured columns."""
    df_base = pd.DataFrame({
        "Price": [10.0, 15.0, 20.0, 25.0, 30.0] * 100,
        "Quantity": [1, 2, 3, 4, 5] * 100
    })
    df_curr = pd.DataFrame({
        "Price": [12.0, 16.0, 22.0, 27.0, 32.0] * 100,
        "Quantity": [1, 2, 3, 4, 5] * 100
    })

    report = generate_drift_report(df_base, df_curr, ["Price", "Quantity"])
    assert len(report) == 2
    assert "psi" in report.columns
    assert "status" in report.columns
    assert "action" in report.columns
