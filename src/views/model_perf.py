"""
Model Performance & Technical Documentation View for RetailPulse Dashboard.
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any


def render_model_performance(meta_seg: Dict[str, Any], churn_metrics: Dict[str, Any]):
    st.title("🔬 Model Performance & Governance")
    st.markdown(
        "Complete technical scorecard and governance details for all analytical models "
        "deployed within the RetailPulse platform."
    )

    t1, t2, t3, t4 = st.tabs([
        "👥 Customer Segmentation",
        "📦 Demand Forecasting",
        "⚠️ Churn & Inactivity",
        "📋 Data Dictionary & Governance"
    ])

    with t1:
        st.subheader("K-Means Customer Segmentation (K=4)")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Silhouette Coefficient", f"{meta_seg.get('silhouette_score', 0.3664):.4f}")
        with m2:
            st.metric("Total Customers Evaluated", f"{meta_seg.get('total_customers', 5878):,}")
        with m3:
            st.metric("2-Component PCA Variance", f"{meta_seg.get('pca_explained_variance_pct', 95.1):.1f}%")

        st.markdown("#### Methodology Details")
        st.markdown(
            "- **Features:** Recency, Frequency, Monetary (Log1p transformed to remove skewness).\n"
            "- **Normalization:** StandardScaler (zero mean, unit variance).\n"
            "- **Algorithm:** K-Means (`k=4, random_state=42, n_init=10`).\n"
            "- **DBSCAN Benchmark:** Evaluated in exploratory phase with `eps=0.5, min_samples=5`. Resulted in 2 clusters with 63 noise points (1.07%). K-Means was selected for operational interpretability."
        )

    with t2:
        st.subheader("Time-Series Demand Forecasting Evaluation")
        st.markdown(
            "- **Evaluation Protocol:** Chronological train/test split (last 3 calendar months held out). Zero random shuffling.\n"
            "- **Champion Model:** Exponential Smoothing (Simple Exponential Smoothing & Holt's Linear Trend).\n"
            "- **Benchmark Baselines:** Naive Forecast (last value) & Rolling 3-Month Average.\n"
            "- **Primary Metric:** WAPE (Weighted Absolute Percentage Error), robust to zero-demand actual months."
        )
        bench_df = pd.DataFrame([
            {"Model Approach": "Naive Baseline", "WAPE (%)": "38.5% - 55.2%", "Complexity": "Minimal", "Suitability": "Benchmark"},
            {"Model Approach": "Rolling 3M Average", "WAPE (%)": "24.1% - 39.8%", "Complexity": "Low", "Suitability": "Short-term smooth demand"},
            {"Model Approach": "Exponential Smoothing", "WAPE (%)": "18.4% - 31.0%", "Complexity": "Moderate", "Suitability": "Champion Model"}
        ])
        st.dataframe(bench_df, use_container_width=True)

    with t3:
        st.subheader("Customer Churn Classification Scorecard")
        st.markdown(
            "- **Observation Cutoff Date:** 2011-09-01 (90 days before final transaction).\n"
            "- **Outcome Horizon:** 90 days.\n"
            "- **Target:** Churned (1) if no purchases in outcome window; Active (0) otherwise.\n"
            "- **Validation:** Stratified 80/20 train/test split on prior customer feature vectors."
        )
        if churn_metrics:
            rf_m = churn_metrics.get("random_forest", {})
            lr_m = churn_metrics.get("baseline_logistic_regression", {})
            churn_comp = pd.DataFrame([
                {"Model": "Logistic Regression (Baseline)", "Precision": lr_m.get("Precision"), "Recall": lr_m.get("Recall"), "F1-Score": lr_m.get("F1_Score"), "ROC-AUC": lr_m.get("ROC_AUC"), "PR-AUC": lr_m.get("PR_AUC")},
                {"Model": "Random Forest (Nonlinear)", "Precision": rf_m.get("Precision"), "Recall": rf_m.get("Recall"), "F1-Score": rf_m.get("F1_Score"), "ROC-AUC": rf_m.get("ROC_AUC"), "PR-AUC": rf_m.get("PR_AUC")}
            ])
            st.dataframe(churn_comp, use_container_width=True)
        else:
            st.info("Churn metrics will populate upon completion of model training.")

    with t4:
        st.subheader("Data Definitions & Pipeline Integrity")
        st.markdown(
            "1. **Recorded Revenue:** `Quantity × Price` across all ledger records.\n"
            "2. **Candidate Sales Revenue:** Standard sales with `Quantity > 0`, `Price > 0`, excluding cancellation invoices (`C` prefix) and bad debt adjustments (`adjust bad debt`).\n"
            "3. **Guest Checkouts:** Records where `Customer ID` is null (preserved in sales analytics, excluded from customer-level RFM).\n"
            "4. **Snapshot Date:** 2011-12-10 (one day following the latest observed transaction)."
        )
        st.markdown("#### Known Operational Limitations")
        st.warning(
            "- **Historical Nature:** Observations are from 2009–2011 UK retail exports; market patterns may differ today.\n"
            "- **Inventory Inferences:** Warehouse stock-on-hand, actual lead times, and stockouts were not recorded.\n"
            "- **Interpretations:** Customer segment labels are provisional business descriptions based on empirical statistics."
        )
