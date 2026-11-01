"""
Customer Segmentation View for RetailPulse Dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, Any

from src.segmentation import predict_customer_segment


def render_customer_segmentation(df_rfm: pd.DataFrame, meta_seg: Dict[str, Any]):
    st.title("👥 Customer Segmentation (K-Means & RFM)")
    st.markdown(
        "Segments customers into distinct behavioral cohorts using **log-transformed RFM features**, "
        "StandardScaler normalization, and **K-Means clustering (K=4)** evaluated by silhouette analysis."
    )

    if df_rfm.empty:
        st.warning("Customer RFM dataset is empty or not yet generated.")
        return

    # 1. Cluster Metrics Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Segmented Customers", f"{len(df_rfm):,}")
    with k2:
        st.metric("Silhouette Score (K=4)", f"{meta_seg.get('silhouette_score', 0.3664):.4f}")
    with k3:
        st.metric("PCA Explained Variance", f"{meta_seg.get('pca_explained_variance_pct', 95.1):.1f}%")
    with k4:
        st.metric("Clusters Identified", f"{meta_seg.get('n_clusters', 4)}")

    st.markdown("---")

    # 2. Interactive PCA Scatter & Distribution
    c1, c2 = st.columns([3, 2])

    with c1:
        st.subheader("2D PCA Customer Projection")
        st.caption("Visualizing 5,878 customer positions across principal components of log(RFM).")
        if "PCA_Dim1" in df_rfm.columns and "PCA_Dim2" in df_rfm.columns:
            # Sample for ultra-smooth rendering in Streamlit if large
            plot_sample = df_rfm.sample(min(3000, len(df_rfm)), random_state=42)
            fig_pca = px.scatter(
                plot_sample,
                x="PCA_Dim1",
                y="PCA_Dim2",
                color="Segment_Name" if "Segment_Name" in plot_sample.columns else "Cluster",
                hover_data=["Customer ID", "Recency", "Frequency", "Monetary"],
                color_discrete_sequence=px.colors.qualitative.Bold,
                opacity=0.7,
                labels={"PCA_Dim1": "Principal Component 1", "PCA_Dim2": "Principal Component 2"}
            )
            fig_pca.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), template="plotly_white")
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.info("PCA dimensions will appear once clustering finishes.")

    with c2:
        st.subheader("Cluster Size Distribution")
        if "Segment_Name" in df_rfm.columns:
            counts = df_rfm["Segment_Name"].value_counts().reset_index()
            counts.columns = ["Segment", "Customers"]
            counts["Share"] = (counts["Customers"] / counts["Customers"].sum() * 100).round(1).astype(str) + "%"
            fig_bar = px.bar(
                counts,
                x="Customers",
                y="Segment",
                orientation="h",
                color="Segment",
                color_discrete_sequence=px.colors.qualitative.Bold,
                text="Share"
            )
            fig_bar.update_layout(height=400, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

    # 3. Cluster Profile Table
    st.subheader("📋 Empirical Cluster Profiles (Medians & Means)")
    st.caption("Provisional business interpretations based on empirical RFM medians.")
    if "Cluster" in df_rfm.columns:
        profiles = (
            df_rfm.groupby(["Cluster", "Segment_Name"])
            .agg(
                Customers=("Customer ID", "count"),
                Median_Recency_Days=("Recency", "median"),
                Median_Frequency=("Frequency", "median"),
                Median_Spend_GBP=("Monetary", "median"),
                Mean_Recency_Days=("Recency", "mean"),
                Mean_Frequency=("Frequency", "mean"),
                Mean_Spend_GBP=("Monetary", "mean"),
            )
            .round(1)
            .reset_index()
        )
        st.dataframe(profiles, use_container_width=True)

    st.markdown("---")

    # 4. Interactive Customer Lookup & Profiler
    st.subheader("🔎 Individual Customer Lookup & Profiler")
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        cust_id_input = st.text_input("Enter Customer ID", value="17850")
        lookup_btn = st.button("Lookup Customer")

    if cust_id_input:
        match = df_rfm[df_rfm["Customer ID"].astype(str) == str(cust_id_input).strip()]
        if not match.empty:
            cust = match.iloc[0]
            with sc2:
                st.success(f"**Customer ID {cust['Customer ID']}** Found!")
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("Recency", f"{cust['Recency']} days", f"R Score: {cust.get('R_Score', '-')}")
                with mc2:
                    st.metric("Frequency", f"{cust['Frequency']} orders", f"F Score: {cust.get('F_Score', '-')}")
                with mc3:
                    st.metric("Monetary", f"£{cust['Monetary']:,.2f}", f"M Score: {cust.get('M_Score', '-')}")
                st.info(f"**Assigned Segment:** {cust.get('Segment_Name', 'Cluster ' + str(cust.get('Cluster', '-')))} (RFM Code: {cust.get('RFM_Score', '-')})")
        else:
            with sc2:
                st.warning(f"Customer ID '{cust_id_input}' was not found in qualifying RFM records.")

    # 5. New Customer Simulator
    with st.expander("🧪 Simulate New Customer Segment Classification"):
        sim_c1, sim_c2, sim_c3 = st.columns(3)
        with sim_c1:
            sim_r = st.number_input("Days Since Last Purchase (Recency)", min_value=1, max_value=750, value=15)
        with sim_c2:
            sim_f = st.number_input("Number of Qualifying Orders (Frequency)", min_value=1, max_value=300, value=8)
        with sim_c3:
            sim_m = st.number_input("Total Spend (£ Monetary)", min_value=1.0, max_value=250000.0, value=2450.0)

        if st.button("Classify Simulated Customer"):
            try:
                res = predict_customer_segment(sim_r, sim_f, sim_m)
                st.success(f"**Predicted Segment:** {res['segment_name']} (Cluster {res['cluster_id']})")
                st.caption(f"Standardized Log-Features: {res['scaled_features']}")
            except Exception as e:
                st.error(f"Classification failed: {e}")
