"""
RetailPulse Product-Level Demand Forecasting Module.
Implements chronological time-series evaluation, naive and rolling average baselines,
Exponential Smoothing forecasting, and forward-looking horizon projections.
"""
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.api import SimpleExpSmoothing, Holt

from src import config
from src.metrics import evaluate_forecast


def prepare_product_series(
    df_monthly_demand: pd.DataFrame,
    stock_code: str,
    fill_missing_months: bool = True
) -> pd.Series:
    """
    Extract a clean, regular monthly time series of units sold for a product.
    Missing calendar months are filled with 0.0 to reflect zero recorded demand.
    """
    prod_data = df_monthly_demand[df_monthly_demand["StockCode"] == str(stock_code)].copy()
    if prod_data.empty:
        raise ValueError(f"No monthly demand records found for StockCode '{stock_code}'")

    prod_data = prod_data.sort_values("Month").set_index("Month")
    series = prod_data["Units_Sold"].astype(float)

    if fill_missing_months:
        full_idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq="MS")
        series = series.reindex(full_idx, fill_value=0.0)

    series.index.name = "Month"
    series.name = f"Units_Sold_{stock_code}"
    return series


def train_test_split_chronological(
    series: pd.Series,
    test_periods: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Split time series chronologically without shuffling.
    Guarantees no future data leaks into the training partition.
    """
    if len(series) <= test_periods:
        raise ValueError(
            f"Series length ({len(series)}) must be greater than test_periods ({test_periods})"
        )
    train = series.iloc[:-test_periods]
    test = series.iloc[-test_periods:]
    return train, test


def forecast_naive(train: pd.Series, test_periods: int) -> pd.Series:
    """Naive baseline: repeats the last observed historical value."""
    last_val = train.iloc[-1]
    test_index = pd.date_range(
        start=train.index[-1] + pd.DateOffset(months=1),
        periods=test_periods,
        freq="MS"
    )
    return pd.Series(np.full(test_periods, last_val), index=test_index, name="Naive_Baseline")


def forecast_rolling_mean(train: pd.Series, test_periods: int, window: int = 3) -> pd.Series:
    """Rolling average baseline: repeats the mean of the last `window` observations."""
    w = min(window, len(train))
    rolling_val = float(train.iloc[-w:].mean())
    test_index = pd.date_range(
        start=train.index[-1] + pd.DateOffset(months=1),
        periods=test_periods,
        freq="MS"
    )
    return pd.Series(np.full(test_periods, rolling_val), index=test_index, name="Rolling_Avg_Baseline")


def fit_exponential_smoothing(train: pd.Series, test_periods: int) -> Tuple[pd.Series, Any]:
    """
    Fit Simple Exponential Smoothing or Holt's Linear Trend (if sufficient data exists)
    with non-negative damping.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(train) >= 8 and (train > 0).sum() >= 4:
            try:
                model = Holt(train, initialization_method="estimated").fit(optimized=True)
                pred = model.forecast(test_periods)
            except Exception:
                model = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
                pred = model.forecast(test_periods)
        else:
            model = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
            pred = model.forecast(test_periods)

    # Unit demand cannot be negative in practice
    pred = pd.Series(np.maximum(pred.values, 0.0), index=pred.index, name="Exponential_Smoothing")
    return pred, model


def evaluate_product_forecasting(
    series: pd.Series,
    test_periods: int = 3
) -> Dict[str, Any]:
    """
    Perform rigorous held-out chronological evaluation comparing:
    - Naive baseline
    - Rolling 3-Month Average baseline
    - Exponential Smoothing model
    """
    train, test = train_test_split_chronological(series, test_periods=test_periods)

    # 1. Predictions
    pred_naive = forecast_naive(train, test_periods)
    pred_rolling = forecast_rolling_mean(train, test_periods, window=3)
    pred_ets, fitted_model = fit_exponential_smoothing(train, test_periods)

    # 2. Evaluation metrics
    metrics_naive = evaluate_forecast(test.values, pred_naive.values)
    metrics_rolling = evaluate_forecast(test.values, pred_rolling.values)
    metrics_ets = evaluate_forecast(test.values, pred_ets.values)

    return {
        "train_len": len(train),
        "test_len": len(test),
        "test_actuals": test.to_dict(),
        "predictions": {
            "Naive": pred_naive.to_dict(),
            "Rolling_3M": pred_rolling.to_dict(),
            "Exponential_Smoothing": pred_ets.to_dict()
        },
        "metrics": {
            "Naive": metrics_naive,
            "Rolling_3M": metrics_rolling,
            "Exponential_Smoothing": metrics_ets
        },
        "best_model": "Exponential_Smoothing" if metrics_ets["WAPE_pct"] <= metrics_rolling["WAPE_pct"] else "Rolling_3M"
    }


def generate_future_forecast(
    series: pd.Series,
    horizon: int = 3,
    model_type: str = "Exponential_Smoothing"
) -> pd.DataFrame:
    """
    Fit on entire historical series and forecast `horizon` months into the future.
    Includes lower and upper uncertainty bounds based on in-sample residual standard deviation.
    """
    future_dates = pd.date_range(
        start=series.index[-1] + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS"
    )

    if model_type == "Naive":
        point_forecast = np.full(horizon, series.iloc[-1])
        residual_std = float(series.diff().dropna().std()) if len(series) > 2 else 1.0
    elif model_type == "Rolling_3M":
        w = min(3, len(series))
        point_forecast = np.full(horizon, series.iloc[-w:].mean())
        residual_std = float(series.rolling(w).mean().dropna().std()) if len(series) > w else 1.0
    else:  # Exponential Smoothing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(series) >= 8:
                try:
                    fit = Holt(series, initialization_method="estimated").fit(optimized=True)
                except Exception:
                    fit = SimpleExpSmoothing(series, initialization_method="estimated").fit(optimized=True)
            else:
                fit = SimpleExpSmoothing(series, initialization_method="estimated").fit(optimized=True)

        point_forecast = np.maximum(fit.forecast(horizon).values, 0.0)
        residuals = series - fit.fittedvalues
        residual_std = float(residuals.std()) if not np.isnan(residuals.std()) else 1.0

    lower_bound = np.maximum(point_forecast - 1.96 * residual_std, 0.0)
    upper_bound = point_forecast + 1.96 * residual_std

    df_forecast = pd.DataFrame({
        "Month": future_dates,
        "Forecast_Units": np.round(point_forecast, 1),
        "Lower_CI_95": np.round(lower_bound, 1),
        "Upper_CI_95": np.round(upper_bound, 1),
        "Model": model_type
    })
    return df_forecast
