"""
RetailPulse Data Loader Module.
Handles loading raw data from the Excel workbook, reading and writing processed
parquet files, providing cached dataset access for the Streamlit dashboard,
and generating test fixtures.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

from src import config
from src.data_processing import (
    clean_transactions,
    build_rfm_table,
    build_monthly_sales_summary,
    build_monthly_product_demand,
    build_product_summary
)


def load_raw_excel(raw_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Read both Year 2009-2010 and Year 2010-2011 sheets from the raw Excel workbook
    and concatenate them into a single raw dataframe.
    """
    path = raw_path or config.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")

    print(f"[DataLoader] Reading Excel file from {path}...")
    sheets_data = []
    for sheet in config.EXCEL_SHEETS:
        print(f"[DataLoader] Loading sheet: {sheet}...")
        df_sheet = pd.read_excel(path, sheet_name=sheet)
        df_sheet["Sheet_Source"] = sheet
        sheets_data.append(df_sheet)

    df_raw = pd.concat(sheets_data, ignore_index=True)
    print(f"[DataLoader] Successfully loaded {len(df_raw):,} raw rows across sheets.")
    return df_raw


def process_and_save_all(raw_path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Run full data cleaning, RFM construction, and demand aggregation,
    saving all outputs to parquet files in data/processed/.
    """
    print("[DataLoader] Commencing full end-to-end data processing pipeline...")
    df_raw = load_raw_excel(raw_path)

    print("[DataLoader] Cleaning transactions and generating audit flags...")
    df_cleaned = clean_transactions(df_raw)

    print("[DataLoader] Building RFM customer table...")
    df_rfm = build_rfm_table(df_cleaned)

    print("[DataLoader] Building monthly sales summary...")
    df_monthly_sales = build_monthly_sales_summary(df_cleaned)

    print("[DataLoader] Building monthly product demand series...")
    df_monthly_demand = build_monthly_product_demand(df_cleaned)

    print("[DataLoader] Building product history and eligibility summary...")
    df_product_summary = build_product_summary(df_monthly_demand)

    # Save to disk
    print(f"[DataLoader] Saving artifacts to {config.PROCESSED_DATA_DIR}...")
    df_cleaned.to_parquet(config.CLEANED_TRANSACTIONS_PARQUET, index=False)
    df_rfm.to_parquet(config.RFM_CUSTOMERS_PARQUET, index=False)
    df_monthly_sales.to_parquet(config.MONTHLY_SALES_PARQUET, index=False)
    df_monthly_demand.to_parquet(config.MONTHLY_DEMAND_PARQUET, index=False)
    df_product_summary.to_parquet(config.PRODUCT_SUMMARY_PARQUET, index=False)

    # Also save a small sample for unit testing
    sample_df = df_cleaned.sample(min(5000, len(df_cleaned)), random_state=config.RANDOM_STATE)
    sample_df.to_parquet(config.SAMPLE_DATA_DIR / "sample_transactions.parquet", index=False)

    print("[DataLoader] All processed datasets generated and saved successfully!")
    return {
        "cleaned_transactions": df_cleaned,
        "rfm_customers": df_rfm,
        "monthly_sales": df_monthly_sales,
        "monthly_demand": df_monthly_demand,
        "product_summary": df_product_summary
    }


def get_dataset(dataset_name: str, force_recompute: bool = False) -> pd.DataFrame:
    """
    Load a processed dataset by name from parquet. If not present or if
    force_recompute is True, run the processing pipeline first.
    """
    mapping = {
        "cleaned_transactions": config.CLEANED_TRANSACTIONS_PARQUET,
        "rfm_customers": config.RFM_CUSTOMERS_PARQUET,
        "monthly_sales": config.MONTHLY_SALES_PARQUET,
        "monthly_demand": config.MONTHLY_DEMAND_PARQUET,
        "product_summary": config.PRODUCT_SUMMARY_PARQUET,
    }

    if dataset_name not in mapping:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(mapping.keys())}")

    target_path = mapping[dataset_name]

    if not force_recompute and target_path.exists():
        return pd.read_parquet(target_path)

    # If any required file is missing, trigger processing
    datasets = process_and_save_all()
    return datasets[dataset_name]


def are_processed_files_ready() -> bool:
    """Check if all key processed parquet files exist on disk."""
    required = [
        config.CLEANED_TRANSACTIONS_PARQUET,
        config.RFM_CUSTOMERS_PARQUET,
        config.MONTHLY_SALES_PARQUET,
        config.MONTHLY_DEMAND_PARQUET,
        config.PRODUCT_SUMMARY_PARQUET,
    ]
    return all(p.exists() for p in required)
