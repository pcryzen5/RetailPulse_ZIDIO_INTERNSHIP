"""
RetailPulse — AI-Powered Customer Analytics & Demand Forecasting Platform.
Main Streamlit Application Entrypoint.
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path

# Configure Streamlit page settings
st.set_page_config(
    page_title="RetailPulse — Customer Analytics & Demand Forecasting",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern executive dashboard aesthetics
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #1e3a8a;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #4b5563;
    }
    /* Sidebar branding */
    .sidebar-brand {
        padding: 1rem 0;
        text-align: center;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .sidebar-brand h2 {
        color: #1e3a8a;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .sidebar-brand p {
        color: #6b7280;
        font-size: 0.85rem;
        margin: 0.2rem 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

from src import config
from src.views.overview import render_executive_overview
from src.views.sales import render_sales_analytics
from src.views.segmentation_view import render_customer_segmentation
from src.views.churn_view import render_churn_analysis
from src.views.forecasting_view import render_demand_forecasting
from src.views.inventory_view import render_inventory_insights
from src.views.model_perf import render_model_performance
from src.views.drift_view import render_drift_monitoring


@st.cache_data(show_spinner=False)
def load_cached_data():
    """
    Load precomputed processed datasets from parquet files for near-instant rendering.
    """
    data = {}
    try:
        if config.CLEANED_TRANSACTIONS_PARQUET.exists():
            data["cleaned"] = pd.read_parquet(config.CLEANED_TRANSACTIONS_PARQUET)
        else:
            data["cleaned"] = pd.DataFrame()

        if config.RFM_CUSTOMERS_PARQUET.exists():
            data["rfm"] = pd.read_parquet(config.RFM_CUSTOMERS_PARQUET)
        else:
            data["rfm"] = pd.DataFrame()

        if config.MONTHLY_SALES_PARQUET.exists():
            data["monthly_sales"] = pd.read_parquet(config.MONTHLY_SALES_PARQUET)
        else:
            data["monthly_sales"] = pd.DataFrame()

        if config.MONTHLY_DEMAND_PARQUET.exists():
            data["monthly_demand"] = pd.read_parquet(config.MONTHLY_DEMAND_PARQUET)
        else:
            data["monthly_demand"] = pd.DataFrame()

        if config.PRODUCT_SUMMARY_PARQUET.exists():
            data["product_summary"] = pd.read_parquet(config.PRODUCT_SUMMARY_PARQUET)
        else:
            data["product_summary"] = pd.DataFrame()

        # Metadata
        if config.SEGMENT_PROFILES_PATH.exists():
            with open(config.SEGMENT_PROFILES_PATH, "r", encoding="utf-8") as f:
                data["meta_seg"] = json.load(f)
        else:
            data["meta_seg"] = {}

        if config.CHURN_METRICS_PATH.exists():
            with open(config.CHURN_METRICS_PATH, "r", encoding="utf-8") as f:
                data["churn_metrics"] = json.load(f)
        else:
            data["churn_metrics"] = {}

    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return {}

    return data


def main():
    # Sidebar branding & navigation
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <h2>🛍️ RetailPulse</h2>
            <p>AI Customer Analytics & Demand Platform</p>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            options=[
                "📈 Executive Overview",
                "📊 Sales Analytics",
                "👥 Customer Segmentation",
                "⚠️ Churn & Inactivity Risk",
                "📦 Demand Forecasting",
                "🏭 Inventory Insights",
                "🛡️ MLOps & Drift Monitoring",
                "🔬 Model Performance & Info"
            ],
            index=0
        )

        st.markdown("---")
        st.markdown(
            "**Dataset:** Online Retail II (UK)\n\n"
            "**Records:** ~1.03M Transactions\n\n"
            "**Customers:** 5,878 Unique\n\n"
            "**Period:** Dec 2009 – Dec 2011"
        )
        st.caption("RetailPulse v1.0 • Internship Capstone")

    # Load data
    data = load_cached_data()

    if not data or data.get("cleaned") is None or data["cleaned"].empty:
        st.warning(
            "⚠️ **Processed datasets not found or still generating.**\n\n"
            "Please ensure the data generation pipeline has executed (`python scripts/build_artifacts.py`)."
        )
        return

    df_cleaned = data["cleaned"]
    df_rfm = data.get("rfm", pd.DataFrame())
    df_monthly_sales = data.get("monthly_sales", pd.DataFrame())
    df_monthly_demand = data.get("monthly_demand", pd.DataFrame())
    df_product_summary = data.get("product_summary", pd.DataFrame())
    meta_seg = data.get("meta_seg", {})
    churn_metrics = data.get("churn_metrics", {})

    # Route navigation
    if page == "📈 Executive Overview":
        render_executive_overview(df_cleaned, df_rfm, df_monthly_sales, meta_seg)
    elif page == "📊 Sales Analytics":
        render_sales_analytics(df_cleaned, df_monthly_sales)
    elif page == "👥 Customer Segmentation":
        render_customer_segmentation(df_rfm, meta_seg)
    elif page == "⚠️ Churn & Inactivity Risk":
        render_churn_analysis(df_rfm, churn_metrics)
    elif page == "📦 Demand Forecasting":
        render_demand_forecasting(df_monthly_demand, df_product_summary)
    elif page == "🏭 Inventory Insights":
        render_inventory_insights(df_monthly_demand, df_product_summary)
    elif page == "🛡️ MLOps & Drift Monitoring":
        render_drift_monitoring(df_cleaned, df_rfm)
    elif page == "🔬 Model Performance & Info":
        render_model_performance(meta_seg, churn_metrics)


if __name__ == "__main__":
    main()
