# RetailPulse — AI-Powered Customer Analytics & Demand Forecasting Platform

[![Python](https://img.shields.io/badge/Python-3.13.5-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-18%20Passed-success.svg)]()

> **RetailPulse** is an end-to-end retail intelligence platform combining **transaction-level audit data engineering**, **RFM customer segmentation**, **leakage-free churn prediction**, **chronological product demand forecasting**, **inventory planning heuristics**, and **MLOps concept drift monitoring** into an interactive Streamlit application.

---

## 🌟 Key Platform Features

1. **Executive Overview & KPI Dashboard:**
   - Real-time aggregation of >1M transactions, 5,878 unique customers, and ~4,900 distinct products.
   - Revenue reconciliation comparing total recorded ledger revenue (£18.1M) with candidate commercial sales (£16.6M).
2. **Sales & Geographic Analytics:**
   - Dynamic monthly trend analysis with multi-axis combo charts (Units Sold vs Revenue).
   - Country-level breakdown and top revenue-generating SKUs with custom date-range filters and CSV export.
3. **Customer RFM Segmentation (K-Means, K=4):**
   - Monotonic $\ln(1+X)$ skewness mitigation and StandardScaler normalization.
   - Interactive 2D PCA customer projection (95.13% explained variance, silhouette score 0.3664).
   - Individual Customer Lookup tool by Customer ID and an interactive Customer Segment Simulator.
4. **Customer Churn & Inactivity Risk Analysis:**
   - Rigorous, leakage-free 90-day forward outcome window (observation cutoff: 2011-09-01).
   - Supervised Machine Learning benchmark: Logistic Regression (ROC-AUC: 0.7985) vs Random Forest (ROC-AUC: 0.8134).
   - Transparent, rule-based operational inactivity risk tiers (Active, Stable, Moderate Risk, High Inactivity, Dormant).
5. **Product-Level Demand Forecasting:**
   - Chronological held-out evaluation (last 3 calendar months held out, zero random shuffling).
   - Benchmark models: Naive baseline, Rolling 3-Month Average, and Exponential Smoothing.
   - Forward-looking 1-to-6 month horizon forecasting with 95% uncertainty confidence intervals.
6. **Demand-Driven Inventory Insights:**
   - User-configurable supplier lead times (weeks) and cycle service levels (90% to 99%).
   - Automated safety stock calculation, suggested reorder points, and demand velocity trend detection.
7. **MLOps & Data Drift Monitoring:**
   - Continuous drift auditing between Year 2009–2010 (baseline) and Year 2010–2011 (production stream) using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests.
   - Interactive Real-Time Concept Drift Simulator with automated model retraining triggers.
7. **Model Performance & Governance Scorecard:**
   - Transparent auditing of all model parameters, benchmark metrics, data dictionaries, and operational caveats.

---

## 🏗️ Architecture & Project Directory Structure

```
RetailPulse_ZIDIO_INTERNSHIP/
├── data/
│   ├── interim/quarantine/
│   │   └── quarantine_batch.csv         # Preserved audit quarantine records
│   ├── processed/                       # Columnar Parquet datasets for sub-second rendering
│   │   ├── cleaned_transactions.parquet
│   │   ├── rfm_customers.parquet
│   │   ├── monthly_sales_summary.parquet
│   │   ├── monthly_product_demand.parquet
│   │   └── product_summary.parquet
│   └── sample/                          # Lightweight sample fixtures for unit testing
├── notebooks/
│   ├── RetailPulse_Day01_EDA.ipynb      # Day 1: Preserved exploratory data analysis
│   ├── RetailPulse_Day_02_and_Day_03.ipynb # Day 2-3: Preserved cleaning & RFM segmentation
│   ├── RetailPulse_Day04_Churn_Analysis.ipynb # Day 4: Leakage-free churn & inactivity analysis
│   ├── RetailPulse_Day05_Demand_Forecasting.ipynb # Day 5: Chronological demand forecasting
│   └── RetailPulse_Model_Training.ipynb # Reusable model training & exporter
├── src/
│   ├── __init__.py
│   ├── config.py                        # Centralized paths and hyperparameters
│   ├── data_processing.py               # Data cleaning, RFM, & demand builders
│   ├── data_loader.py                   # Raw Excel ingestion, Parquet I/O, & caching
│   ├── segmentation.py                  # RFM scaling, K-Means clustering, PCA, & inference
│   ├── churn.py                         # 90-day window classification & risk tiers
│   ├── forecasting.py                   # Exponential smoothing, baselines, & horizons
│   ├── inventory.py                     # Safety stock, reorder point, & trend logic
│   ├── metrics.py                       # MAE, RMSE, WAPE, and safe MAPE calculations
│   └── views/                           # Modular Streamlit page renderers
├── models/
│   ├── segmentation/                    # Serialized KMeans (K=4), scaler, and profiles
│   ├── churn/                           # Serialized Random Forest classifier & metrics
│   └── forecasting/                     # Cross-product forecasting benchmark results
├── tests/                               # Comprehensive automated test suite (14 tests)
├── app.py                               # Interactive Streamlit application entrypoint
├── requirements.txt                     # Pinned project dependencies
└── README.md
```

---

## 🚀 Setup & Execution Instructions

### 1. Environment Setup (Windows 11 PowerShell)

Ensure Python 3.13 is available, then activate the local virtual environment:

```powershell
# Navigate to project directory
cd c:\Users\shubhAM\Desktop\RetailPulse_ZIDIO_INTERNSHIP

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

*(Optional) If setting up from scratch on a new machine:*
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Data Processing & Model Artifact Generation

The processed Parquet datasets and serialized machine learning models are already pre-generated and verified. To re-generate them from the raw Excel workbook:

```powershell
python scripts/build_artifacts.py
python scripts/train_churn.py
python scripts/train_forecasting.py
```

### 3. Launch the Streamlit Dashboard

Start the interactive web dashboard with a single command:

```powershell
streamlit run app.py
```

Once running, open your browser and navigate to:
```
http://localhost:8501
```

### 4. Run Automated Pytest Suite

To verify data integrity, model loading, and business logic:

```powershell
python -m pytest -v
```

**Expected Outcome:** `14 passed in ~8s (100% pass rate)`.

---

## 📊 Measured Model Performance Summary

| Analytical Domain | Model / Method | Key Metrics Evaluated |
|---|---|---|
| **Customer Segmentation** | K-Means ($K=4$) on Log(RFM) + StandardScaler | **Silhouette Score:** 0.3664<br>**PCA Variance:** 95.13%<br>**Customers:** 5,878 |
| **Customer Churn (90-Day)** | Random Forest Classifier | **ROC-AUC:** 0.8134<br>**PR-AUC:** 0.8255<br>**F1-Score:** 0.7869 |
| **Churn Baseline** | Logistic Regression | **ROC-AUC:** 0.7985<br>**PR-AUC:** 0.8225<br>**F1-Score:** 0.7676 |
| **Demand Forecasting** | Exponential Smoothing (SES / Holt) | Chronological held-out validation (WAPE robust to zero-demand periods) |
| **Forecast Baselines** | Naive & Rolling 3-Month Average | Benchmark baselines for intermittent retail demand |

---

## ⚠️ Known Operational Scope & Limitations

1. **Historical Observation Window:** Data represents UK retail exports from December 2009 to December 2011. Economic dynamics and seasonality reflect that specific timeframe.
2. **Inventory Guidance Heuristics:** Warehouse stock-on-hand, true vendor lead times, and out-of-stock occurrences were not captured in the transactional ledger. Inventory recommendations are demand-driven planning heuristics.
3. **Provisional Persona Labels:** Cluster designations (High-Value Champions, At-Risk High Value, Dormant) represent operational business interpretations based on empirical RFM medians.

---

## 📄 License & Attribution

- **Project:** RetailPulse Internship Capstone Project.
- **Dataset:** Online Retail II Dataset (UCI Machine Learning Repository / Kaggle).
