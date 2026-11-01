"""
RetailPulse Data Processing Module.
Implements the exact data cleaning, transaction-quality classification,
description recovery, RFM feature engineering, and demand aggregation
established in the exploratory analysis notebooks.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional


def clean_transactions(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw transaction data and compute audit flags.

    Steps:
    1. Calculate transaction revenue = Quantity * Price.
    2. Deduplicate exact duplicate rows.
    3. Generate quality classification flags.
    4. Perform unambiguous description recovery using StockCode mapping.
    5. Construct Candidate_Sales_Revenue for standard sales.
    """
    df = df_raw.copy()

    # 1. Ensure Revenue column
    if "Revenue" not in df.columns:
        df["Revenue"] = df["Quantity"] * df["Price"]

    # Normalize mixed-type columns to string to guarantee clean Parquet serialization
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Country"] = df["Country"].astype(str).str.strip()

    # 2. Exact duplicate removal
    df = df.drop_duplicates().copy()

    # Ensure InvoiceDate is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # 3. Transaction quality flags
    df["Is_Cancellation_Invoice"] = df["Invoice"].str.startswith("C")
    df["Is_Negative_Quantity"] = df["Quantity"] < 0
    df["Is_Zero_Price"] = df["Price"] == 0
    df["Is_Negative_Price"] = df["Price"] < 0
    df["Has_Customer_ID"] = df["Customer ID"].notna()

    # Flag accounting adjustments
    df["Is_Bad_Debt_Adjustment"] = (
        df["Description"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq("adjust bad debt")
        .fillna(False)
    )

    # Standard priced sale: positive qty, positive price, not bad debt, not cancellation
    df["Is_Standard_Priced_Sale"] = (
        (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & (~df["Is_Bad_Debt_Adjustment"])
        & (~df["Is_Cancellation_Invoice"])
    )

    df["Is_Cancellation_Or_Reversal"] = (
        df["Is_Cancellation_Invoice"] & (df["Quantity"] < 0)
    )

    df["Is_Zero_Price_Adjustment"] = (
        (df["Price"] == 0) | (df["Quantity"] == 0)
    )

    df["Is_Unusual_Cancellation_Record"] = (
        df["Is_Cancellation_Invoice"] & (df["Quantity"] > 0)
    )

    # 4. Description recovery for missing descriptions with single known mapping
    desc_mapping = (
        df.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(
            Unique_Descriptions="nunique",
            Most_Common_Description=lambda s: s.mode().iloc[0]
        )
    )

    single_desc_codes = desc_mapping[desc_mapping["Unique_Descriptions"] == 1]["Most_Common_Description"]
    missing_desc_mask = df["Description"].isna()
    recoverable_mask = missing_desc_mask & df["StockCode"].isin(single_desc_codes.index)
    df.loc[recoverable_mask, "Description"] = df.loc[recoverable_mask, "StockCode"].map(single_desc_codes)

    # 5. Ensure Description and Customer ID have clean types for serialization
    df["Description"] = df["Description"].fillna("Unknown").astype(str)
    df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")

    # 6. Candidate sales revenue
    df["Candidate_Sales_Revenue"] = df["Revenue"].where(df["Is_Standard_Priced_Sale"], 0.0)

    return df


def build_rfm_table(df_analysis: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Construct customer-level RFM features from qualifying transactions.
    
    Qualifying transaction criteria:
    - Customer ID is not null
    - Quantity > 0
    - Price > 0
    - Not a cancellation invoice
    - Not a bad debt adjustment
    """
    rfm_tx = df_analysis.loc[
        (df_analysis["Customer ID"].notna())
        & (df_analysis["Quantity"] > 0)
        & (df_analysis["Price"] > 0)
        & (~df_analysis["Is_Cancellation_Invoice"])
        & (~df_analysis["Is_Bad_Debt_Adjustment"])
    ].copy()

    # Format Customer ID as string identifier
    rfm_tx["Customer ID"] = rfm_tx["Customer ID"].astype("Int64").astype(str)

    if snapshot_date is None:
        snapshot_date = rfm_tx["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    rfm = (
        rfm_tx.groupby("Customer ID")
        .agg(
            Last_Purchase_Date=("InvoiceDate", "max"),
            First_Purchase_Date=("InvoiceDate", "min"),
            Frequency=("Invoice", "nunique"),
            Monetary=("Revenue", "sum"),
            Total_Units=("Quantity", "sum")
        )
        .reset_index()
    )

    rfm["Recency"] = (snapshot_date - rfm["Last_Purchase_Date"].dt.normalize()).dt.days

    # Customer tenure (days from first purchase to snapshot)
    rfm["Tenure_Days"] = (snapshot_date - rfm["First_Purchase_Date"].dt.normalize()).dt.days

    # Quintile scores (1 to 5)
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"].rank(method="first"),
        q=5,
        labels=[5, 4, 3, 2, 1]
    ).astype(int)

    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    def assign_segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r <= 2 and f >= 4:
            return "At Risk - Loyal Customers"
        elif r >= 4 and f <= 2:
            return "Recent - Need Engagement"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Lost Customers"
        elif m >= 4:
            return "High Spenders"
        elif f >= 3:
            return "Regular Repeat Customers"
        else:
            return "Average / Occasional"

    rfm["Rule_Based_Segment"] = rfm.apply(assign_segment, axis=1)

    return rfm


def build_monthly_sales_summary(df_analysis: pd.DataFrame) -> pd.DataFrame:
    """
    Construct monthly aggregated revenue series comparing recorded revenue
    and candidate sales revenue.
    """
    df_copy = df_analysis.copy()
    df_copy["Month"] = df_copy["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    monthly_sales = (
        df_copy.groupby("Month")
        .agg(
            Recorded_Revenue=("Revenue", "sum"),
            Candidate_Sales_Revenue=("Candidate_Sales_Revenue", "sum"),
            Transaction_Rows=("Invoice", "size"),
            Candidate_Sale_Rows=("Is_Standard_Priced_Sale", "sum"),
            Distinct_Customers=("Customer ID", "nunique"),
            Distinct_Products=("StockCode", "nunique")
        )
        .reset_index()
    )
    return monthly_sales


def build_monthly_product_demand(df_analysis: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly units sold and revenue by product.
    Only includes standard sales transactions.
    """
    df_demand = df_analysis.loc[
        (df_analysis["Quantity"] > 0)
        & (df_analysis["Price"] > 0)
        & (~df_analysis["Is_Cancellation_Invoice"])
        & (~df_analysis["Is_Bad_Debt_Adjustment"])
    ].copy()

    df_demand["Month"] = df_demand["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    # Top description for each StockCode
    stock_desc = (
        df_demand.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(lambda s: s.mode().iloc[0] if not s.empty else "Unknown")
        .to_dict()
    )

    monthly_demand = (
        df_demand.groupby(["StockCode", "Month"], as_index=False)
        .agg(
            Units_Sold=("Quantity", "sum"),
            Sales_Revenue=("Revenue", "sum"),
            Transaction_Rows=("Invoice", "size")
        )
    )

    monthly_demand["Description"] = monthly_demand["StockCode"].map(stock_desc).fillna("Unknown")

    # Add 3-month rolling statistics per product
    monthly_demand = monthly_demand.sort_values(["StockCode", "Month"])
    monthly_demand["Rolling_3M_Avg_Units"] = (
        monthly_demand.groupby("StockCode")["Units_Sold"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )
    monthly_demand["Rolling_3M_Std_Units"] = (
        monthly_demand.groupby("StockCode")["Units_Sold"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).std().fillna(0.0))
    )

    return monthly_demand


def build_product_summary(monthly_demand: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize product historical activity, eligibility for time-series forecasting,
    and average monthly demand.
    """
    product_summary = (
        monthly_demand.groupby("StockCode", as_index=False)
        .agg(
            Description=("Description", "first"),
            Total_Units_Sold=("Units_Sold", "sum"),
            Total_Sales_Revenue=("Sales_Revenue", "sum"),
            Active_Months=("Month", "nunique"),
            First_Month=("Month", "min"),
            Last_Month=("Month", "max"),
            Mean_Monthly_Units=("Units_Sold", "mean"),
            Std_Monthly_Units=("Units_Sold", "std")
        )
    )
    product_summary["Std_Monthly_Units"] = product_summary["Std_Monthly_Units"].fillna(0.0)
    product_summary["Is_Forecasting_Eligible"] = product_summary["Active_Months"] >= 18

    product_summary = product_summary.sort_values(
        ["Active_Months", "Total_Units_Sold"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return product_summary
