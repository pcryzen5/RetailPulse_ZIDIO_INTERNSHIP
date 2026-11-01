"""
Executive Overview View for RetailPulse Dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any


def render_executive_overview(
    df_cleaned: pd.DataFrame,
    df_rfm: pd.DataFrame,
    df_monthly_sales: pd.DataFrame,
    meta_seg: Dict[str, Any]
):
    st.title("📈 Executive Overview")
    st.markdown(
        "Welcome to **RetailPulse** — AI-Powered Customer Analytics & Demand Forecasting Platform. "
        "Here is the high-level operational and business pulse computed from the historical retail transactions."
    )

    # 1. Top-Level Metric Cards
    total_records = len(df_cleaned)
    total_customers = df_rfm["Customer ID"].nunique() if not df_rfm.empty else 0
    total_products = df_cleaned["StockCode"].nunique()
    recorded_rev = df_cleaned["Revenue"].sum()
    candidate_sales_rev = df_cleaned["Candidate_Sales_Revenue"].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Transactions", f"{total_records:,}")
    with col2:
        st.metric("Unique Customers", f"{total_customers:,}")
    with col3:
        st.metric("Unique Products", f"{total_products:,}")
    with col4:
        st.metric("Recorded Revenue", f"£{recorded_rev:,.0f}")
    with col5:
        st.metric("Candidate Sales", f"£{candidate_sales_rev:,.0f}")

    st.markdown("---")

    # 2. Revenue Trend & Customer Segments
    c1, c2 = st.columns([3, 2])

    with c1:
        st.subheader("Monthly Revenue Trajectory (2009 - 2011)")
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(
            x=df_monthly_sales["Month"],
            y=df_monthly_sales["Recorded_Revenue"],
            mode="lines+markers",
            name="Recorded Revenue",
            line=dict(color="#1f77b4", width=2.5)
        ))
        fig_rev.add_trace(go.Scatter(
            x=df_monthly_sales["Month"],
            y=df_monthly_sales["Candidate_Sales_Revenue"],
            mode="lines+markers",
            name="Candidate Sales Revenue",
            line=dict(color="#2ca02c", width=2.5, dash="dash")
        ))
        fig_rev.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Month",
            yaxis_title="Revenue (£)",
            template="plotly_white"
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    with c2:
        st.subheader("Customer Segment Distribution")
        if "Segment_Name" in df_rfm.columns:
            seg_counts = df_rfm["Segment_Name"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            fig_pie = px.pie(
                seg_counts,
                names="Segment",
                values="Count",
                color_discrete_sequence=px.colors.qualitative.Prism,
                hole=0.4
            )
            fig_pie.update_layout(
                height=360,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Segment distributions pending model export.")

    # 3. Quick System Health & Data Integrity Callout
    st.markdown("### 🔍 Data Integrity & Reconciliation Summary")
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        cancellations = df_cleaned["Is_Cancellation_Invoice"].sum()
        st.info(f"**Cancellations Preserved:** {cancellations:,} records ({cancellations/total_records*100:.1f}%)")
    with ic2:
        missing_cust = df_cleaned["Customer ID"].isna().sum()
        st.info(f"**Guest / Unidentified Checkouts:** {missing_cust:,} records ({missing_cust/total_records*100:.1f}%)")
    with ic3:
        sil_score = meta_seg.get("silhouette_score", "N/A")
        st.success(f"**Segmentation Quality:** K=4 Silhouette Score = **{sil_score}**")
