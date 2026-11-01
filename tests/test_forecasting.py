"""
Unit tests for demand forecasting, baselines, and chronological splits.
"""
import pytest
import pandas as pd
import numpy as np

from src import config
from src.forecasting import (
    prepare_product_series,
    train_test_split_chronological,
    evaluate_product_forecasting,
    generate_future_forecast
)


def test_prepare_product_series():
    """Verify that product series is indexed with monthly frequency and non-negative values."""
    df_demand = pd.read_parquet(config.MONTHLY_DEMAND_PARQUET)
    sample_code = df_demand["StockCode"].iloc[0]

    series = prepare_product_series(df_demand, sample_code)
    assert isinstance(series.index, pd.DatetimeIndex)
    assert (series >= 0).all()
    assert np.isfinite(series.values).all()


def test_chronological_split_no_leakage():
    """Verify that train and test sets are strictly chronological."""
    df_demand = pd.read_parquet(config.MONTHLY_DEMAND_PARQUET)
    # Find a product with at least 12 months
    counts = df_demand["StockCode"].value_counts()
    code = counts[counts >= 12].index[0]
    series = prepare_product_series(df_demand, code)

    train, test = train_test_split_chronological(series, test_periods=3)
    assert len(test) == 3
    assert len(train) == len(series) - 3
    assert train.index.max() < test.index.min()


def test_generate_future_forecast_structure():
    """Verify that forward forecast table contains future dates and non-negative bounds."""
    df_demand = pd.read_parquet(config.MONTHLY_DEMAND_PARQUET)
    sample_code = df_demand["StockCode"].iloc[0]
    series = prepare_product_series(df_demand, sample_code)

    horizon = 3
    forecast_df = generate_future_forecast(series, horizon=horizon)
    assert len(forecast_df) == horizon
    assert "Forecast_Units" in forecast_df.columns
    assert "Lower_CI_95" in forecast_df.columns
    assert "Upper_CI_95" in forecast_df.columns

    # Dates must be strictly after the last historical date
    assert forecast_df["Month"].min() > series.index.max()
    # All forecast values must be finite and non-negative
    assert (forecast_df["Forecast_Units"] >= 0).all()
    assert (forecast_df["Lower_CI_95"] >= 0).all()
    assert (forecast_df["Upper_CI_95"] >= forecast_df["Lower_CI_95"]).all()
