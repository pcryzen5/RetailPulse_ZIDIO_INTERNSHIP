"""
Unit tests for data loader, data cleaning, and RFM calculations.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src import config
from src.data_processing import clean_transactions, build_rfm_table
from src.data_loader import get_dataset, are_processed_files_ready


def test_processed_files_exist():
    """Verify that all key parquet files were generated and exist on disk."""
    assert are_processed_files_ready() is True
    assert config.CLEANED_TRANSACTIONS_PARQUET.exists()
    assert config.RFM_CUSTOMERS_PARQUET.exists()
    assert config.MONTHLY_SALES_PARQUET.exists()
    assert config.MONTHLY_DEMAND_PARQUET.exists()
    assert config.PRODUCT_SUMMARY_PARQUET.exists()


def test_cleaned_transactions_columns_and_types():
    """Verify required columns and data types in cleaned transactions."""
    df = pd.read_parquet(config.CLEANED_TRANSACTIONS_PARQUET)
    required_cols = [
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID", "Country",
        "Revenue", "Is_Cancellation_Invoice", "Is_Standard_Priced_Sale",
        "Candidate_Sales_Revenue"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert len(df) > 1000000, f"Expected >1M records, found {len(df)}"


def test_revenue_recalculation_consistency():
    """Verify that Revenue strictly equals Quantity * Price across all transactions."""
    # Test on a deterministic sample for speed
    df = pd.read_parquet(config.CLEANED_TRANSACTIONS_PARQUET)
    sample = df.head(10000)
    calculated_rev = sample["Quantity"] * sample["Price"]
    assert np.allclose(sample["Revenue"], calculated_rev, atol=1e-3)


def test_rfm_table_integrity():
    """Verify that RFM table contains no missing customer IDs, correct columns, and valid R/F/M ranges."""
    df_rfm = pd.read_parquet(config.RFM_CUSTOMERS_PARQUET)
    assert len(df_rfm) == 5878, f"Expected 5,878 customers, got {len(df_rfm)}"
    assert df_rfm["Customer ID"].isna().sum() == 0
    assert df_rfm["Customer ID"].nunique() == 5878

    # All recency and frequency values must be positive
    assert (df_rfm["Recency"] >= 0).all()
    assert (df_rfm["Frequency"] >= 1).all()
    assert (df_rfm["Monetary"] > 0).all()

    # Scores must be within 1-5
    for s in ["R_Score", "F_Score", "M_Score"]:
        assert df_rfm[s].min() >= 1
        assert df_rfm[s].max() <= 5
