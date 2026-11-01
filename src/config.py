"""
RetailPulse Configuration Module.
Defines project paths, dataset filenames, and core constants.
All paths are resolved relative to the project root directory.
"""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = PROJECT_ROOT / "online_retail_II.xlsx"
INTERIM_DATA_DIR = DATA_DIR / "interim"
QUARANTINE_PATH = INTERIM_DATA_DIR / "quarantine" / "quarantine_batch.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Processed dataset filepaths (stored as fast, schema-preserved parquet)
CLEANED_TRANSACTIONS_PARQUET = PROCESSED_DATA_DIR / "cleaned_transactions.parquet"
RFM_CUSTOMERS_PARQUET = PROCESSED_DATA_DIR / "rfm_customers.parquet"
MONTHLY_DEMAND_PARQUET = PROCESSED_DATA_DIR / "monthly_product_demand.parquet"
PRODUCT_SUMMARY_PARQUET = PROCESSED_DATA_DIR / "product_summary.parquet"
MONTHLY_SALES_PARQUET = PROCESSED_DATA_DIR / "monthly_sales_summary.parquet"

# Models directories
MODELS_DIR = PROJECT_ROOT / "models"
SEGMENTATION_MODEL_DIR = MODELS_DIR / "segmentation"
CHURN_MODEL_DIR = MODELS_DIR / "churn"
FORECASTING_MODEL_DIR = MODELS_DIR / "forecasting"

# Model artifact paths
RFM_SCALER_PATH = SEGMENTATION_MODEL_DIR / "rfm_scaler.joblib"
KMEANS_MODEL_PATH = SEGMENTATION_MODEL_DIR / "kmeans_k4.joblib"
SEGMENT_PROFILES_PATH = SEGMENTATION_MODEL_DIR / "segment_profiles.json"

CHURN_MODEL_PATH = CHURN_MODEL_DIR / "churn_classifier.joblib"
CHURN_METRICS_PATH = CHURN_MODEL_DIR / "churn_metrics.json"

FORECAST_METRICS_PATH = FORECASTING_MODEL_DIR / "forecast_metrics.json"

# Notebooks and docs directories
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"

# Excel sheets
EXCEL_SHEETS = ["Year 2009-2010", "Year 2010-2011"]

# Modeling and segmentation constants
RANDOM_STATE = 42
KMEANS_CLUSTERS = 4
CLUSTERING_FEATURES = ["Recency", "Frequency", "Monetary"]

# Provisional business names for K-Means clusters (K=4)
# (Mapped based on empirical RFM profiles: High Value/Champions, Loyal/Frequent, At Risk, Dormant/Lost)
SEGMENT_NAMES = {
    0: "Loyal Customers",
    1: "Dormant / Lost",
    2: "At Risk High-Spenders",
    3: "High-Value Champions"
}

# Ensure required directories exist
for path in [
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR,
    SEGMENTATION_MODEL_DIR,
    CHURN_MODEL_DIR,
    FORECASTING_MODEL_DIR,
    DOCS_DIR
]:
    path.mkdir(parents=True, exist_ok=True)
