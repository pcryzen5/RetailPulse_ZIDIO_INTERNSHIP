"""
Demand-Based Inventory Insights View for RetailPulse Dashboard.
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any

from src.inventory import (
    calculate_product_inventory_insights,
    build_inventory_overview_table
)


def render_inventory_insights(df_monthly_demand: pd.DataFrame, df_product_summary: pd.DataFrame):
    st.title("🏭 Demand-Based Inventory Insights")
    st.markdown(
        "Translates demand volatility and forecasting into **suggested reorder points**, "
        "**buffer stock requirements**, and **demand velocity trends** under explicit, transparent assumptions."
    )

    if df_monthly_demand.empty:
        st.warning("Monthly demand data not available.")
        return

    # User-Configurable Assumptions
    with st.expander("⚙️ Inventory Planning Assumptions & Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            lead_time = st.slider("Supplier Lead Time (Weeks)", min_value=1.0, max_value=8.0, value=2.0, step=0.5)
        with c2:
            service_level_label = st.selectbox(
                "Target Cycle Service Level",
                options=["90% (Z=1.28)", "95% (Z=1.65)", "97.5% (Z=1.96)", "99% (Z=2.33)"],
                index=1
            )
            z_map = {"90% (Z=1.28)": 1.28, "95% (Z=1.65)": 1.65, "97.5% (Z=1.96)": 1.96, "99% (Z=2.33)": 2.33}
            service_z = z_map[service_level_label]
        with c3:
            st.info("💡 **Formula:** Reorder Point = Lead Time Demand + Safety Stock (Z × σ × √L)")

    # Single Product Deep-Dive
    st.subheader("🔍 Product Inventory Planning Card")

    top_codes = df_product_summary.head(30)["StockCode"].tolist()
    options = {
        f"{row['StockCode']} - {row['Description'][:35]}": row['StockCode']
        for _, row in df_product_summary.head(30).iterrows()
    }

    pc1, pc2 = st.columns([2, 1])
    with pc1:
        sel_label = st.selectbox("Select Product", options=list(options.keys()), index=0)
        sel_code = options[sel_label]
    with pc2:
        user_stock = st.number_input("Current On-Hand Stock (Optional)", min_value=0, value=120, step=10)

    # Calculate insights
    try:
        insights = calculate_product_inventory_insights(
            df_monthly_demand=df_monthly_demand,
            stock_code=sel_code,
            lead_time_weeks=lead_time,
            service_level_z=service_z,
            current_stock_on_hand=user_stock
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Demand Velocity", insights["demand_trend"])
        with m2:
            st.metric("Recent 3M Avg Monthly", f"{insights['recent_3m_avg_monthly_units']} units")
        with m3:
            st.metric("Est. Safety Stock", f"{insights['estimated_safety_stock']} units")
        with m4:
            st.metric("Suggested Reorder Point", f"{insights['suggested_reorder_point']} units")

        # Action callout
        st.success(f"**Stock Assessment:** {insights['stock_status']} — {insights['action_recommendation']}")

    except Exception as e:
        st.error(f"Could not compute insights for {sel_code}: {e}")

    st.markdown("---")

    # Multi-product inventory overview table
    st.subheader("📊 Multi-Product Inventory Planning Table")
    st.caption("Top 25 active retail products evaluated under current lead time and service level settings.")

    overview_df = build_inventory_overview_table(
        df_monthly_demand=df_monthly_demand,
        lead_time_weeks=lead_time,
        service_level_z=service_z,
        top_n=25
    )
    st.dataframe(overview_df, use_container_width=True)

    # Explicit operational disclaimer
    st.warning(
        "📌 **Operational Integrity Note:**\n"
        "The source dataset contains recorded transactions, not live warehouse inventory or ERP supplier catalogs. "
        "These figures serve as demand-driven planning heuristics and should not be represented as verified optimal inventory policies."
    )
