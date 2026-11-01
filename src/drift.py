"""
RetailPulse Data & Concept Drift Monitoring Module.
Implements Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests
to detect distribution shifts between baseline reference data and incoming inference batches.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_bins: int = 10, epsilon: float = 1e-4) -> float:
    """
    Calculate the Population Stability Index (PSI) between baseline and current distributions.
    Interpretation:
    - PSI < 0.10: Insignificant drift (distributions are stable)
    - 0.10 <= PSI < 0.25: Moderate drift (warning; monitor closely)
    - PSI >= 0.25: Significant drift (action required; trigger model retraining)
    """
    baseline_clean = np.asarray(baseline, dtype=float)
    baseline_clean = baseline_clean[~np.isnan(baseline_clean)]
    current_clean = np.asarray(current, dtype=float)
    current_clean = current_clean[~np.isnan(current_clean)]

    if len(baseline_clean) == 0 or len(current_clean) == 0:
        return 0.0

    # Quantile bins based on baseline distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(baseline_clean, percentiles)
    # Deduplicate bin edges if data has repeated values
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate frequency proportions per bin
    baseline_counts, _ = np.histogram(baseline_clean, bins=bin_edges)
    current_counts, _ = np.histogram(current_clean, bins=bin_edges)

    baseline_pct = (baseline_counts / len(baseline_clean)) + epsilon
    current_pct = (current_counts / len(current_clean)) + epsilon

    # Normalize after adding epsilon
    baseline_pct /= np.sum(baseline_pct)
    current_pct /= np.sum(current_pct)

    # PSI calculation formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    psi_value = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(np.round(psi_value, 4))


def calculate_ks_test(baseline: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    """
    Perform two-sample Kolmogorov-Smirnov test to detect distribution shift.
    Null hypothesis (H0): Both samples are drawn from the same continuous distribution.
    If p-value < 0.05, we reject H0 -> statistically significant drift detected.
    """
    baseline_clean = np.asarray(baseline, dtype=float)
    baseline_clean = baseline_clean[~np.isnan(baseline_clean)]
    current_clean = np.asarray(current, dtype=float)
    current_clean = current_clean[~np.isnan(current_clean)]

    if len(baseline_clean) == 0 or len(current_clean) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    res = ks_2samp(baseline_clean, current_clean)
    return {
        "statistic": float(np.round(res.statistic, 4)),
        "p_value": float(np.round(res.pvalue, 6)),
        "drift_detected": bool(res.pvalue < 0.05)
    }


def evaluate_feature_drift(baseline: np.ndarray, current: np.ndarray, feature_name: str) -> Dict[str, Any]:
    """
    Compute comprehensive drift scorecard for a numerical feature.
    Combines PSI and KS test with baseline and current distribution statistics.
    """
    psi = calculate_psi(baseline, current)
    ks = calculate_ks_test(baseline, current)

    # Determine overall drift tier
    if psi >= 0.25 or (ks["drift_detected"] and psi >= 0.15):
        status = "Critical Drift"
        action = "Trigger automated model retraining"
        color = "#d62728"
    elif psi >= 0.10 or ks["drift_detected"]:
        status = "Moderate Drift"
        action = "Flag for monitoring; schedule review"
        color = "#ff7f0e"
    else:
        status = "Stable (No Drift)"
        action = "No action required; model within tolerance"
        color = "#2ca02c"

    b_clean = np.asarray(baseline, dtype=float)
    b_clean = b_clean[~np.isnan(b_clean)]
    c_clean = np.asarray(current, dtype=float)
    c_clean = c_clean[~np.isnan(c_clean)]

    return {
        "feature": feature_name,
        "psi": psi,
        "ks_statistic": ks["statistic"],
        "p_value": ks["p_value"],
        "status": status,
        "action": action,
        "color": color,
        "baseline_mean": float(np.round(np.mean(b_clean), 2)) if len(b_clean) else 0.0,
        "current_mean": float(np.round(np.mean(c_clean), 2)) if len(c_clean) else 0.0,
        "baseline_std": float(np.round(np.std(b_clean), 2)) if len(b_clean) else 0.0,
        "current_std": float(np.round(np.std(c_clean), 2)) if len(c_clean) else 0.0
    }


def generate_drift_report(
    df_baseline: pd.DataFrame,
    df_current: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Generate multi-feature drift audit table comparing baseline and current dataframes.
    """
    reports = []
    for col in feature_cols:
        if col in df_baseline.columns and col in df_current.columns:
            rep = evaluate_feature_drift(df_baseline[col].values, df_current[col].values, col)
            reports.append(rep)
    return pd.DataFrame(reports)
