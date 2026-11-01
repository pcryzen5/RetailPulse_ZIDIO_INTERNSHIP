"""
RetailPulse Metrics Module.
Provides standardized calculation functions for forecasting, classification,
and clustering evaluation.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Union


def mean_absolute_error(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Calculate Mean Absolute Error."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return float(np.mean(np.abs(y_t - y_p)))


def root_mean_squared_error(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Calculate Root Mean Squared Error."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return float(np.sqrt(np.mean(np.square(y_t - y_p))))


def weighted_absolute_percentage_error(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Weighted Absolute Percentage Error (WAPE).
    WAPE = sum(|actual - pred|) / sum(actual) * 100
    Safe for series containing zero actuals as long as total actuals > 0.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    total_actual = float(np.sum(np.abs(y_t)))
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_t - y_p)) / total_actual * 100.0)


def safe_mape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series], epsilon: float = 1e-5) -> float:
    """
    Calculate Mean Absolute Percentage Error with zero-actual guard.
    Rows with actual == 0 are excluded or stabilized.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    mask = y_t > epsilon
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100.0)


def evaluate_forecast(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Compute comprehensive forecast evaluation metrics:
    MAE, RMSE, WAPE, and safe MAPE.
    """
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(root_mean_squared_error(y_true, y_pred), 3),
        "WAPE_pct": round(weighted_absolute_percentage_error(y_true, y_pred), 2),
        "MAPE_pct": round(safe_mape(y_true, y_pred), 2)
    }
