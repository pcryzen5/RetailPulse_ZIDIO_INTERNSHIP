"""
Sales Analytics View for RetailPulse Dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_sales_analytics(df_cleaned: pd.DataFrame, df_monthly_sales: pd.DataFrame):
    st.title("📊 Sales & Revenue Analytics")
    st.markdown(
        "Explore monthly transactional trends, country breakdowns, product velocity, "
        "and revenue reconciliation between recorded ledger revenue and candidate standard product sales."
    )

    # 1. Filters in expander
    with st.expander("🛠️ Filter Transactions", expanded=True):
        fc1, fc2 = st.columns(2)
        with fc1:
            all_countries = ["All"] + sorted(df_cleaned["Country"].dropna().unique().tolist())
            selected_country = st.selectbox("Select Country", options=all_countries, index=0)
        with fc2:
            min_date = df_cleaned["InvoiceDate"].min().date()
            max_date = df_cleaned["InvoiceDate"].max().date()
            date_range = st.date_input("Invoice Date Range", value=(min_date, max_date))

    # Apply filters
    filtered_df = df_cleaned.copy()
    if selected_country != "All":
        filtered_df = filtered_df[filtered_df["Country"] == selected_country]

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        filtered_df = filtered_df[(filtered_df["InvoiceDate"] >= start_d) & (filtered_df["InvoiceDate"] < end_d)]

    if filtered_df.empty:
        st.warning("No transactions match the selected filter criteria. Please broaden your selection.")
        return

    # Filtered KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Filtered Records", f"{len(filtered_df):,}")
    with k2:
        st.metric("Recorded Revenue", f"£{filtered_df['Revenue'].sum():,.0f}")
    with k3:
        st.metric("Standard Sales Rev", f"£{filtered_df['Candidate_Sales_Revenue'].sum():,.0f}")
    with k4:
        st.metric("Active Products", f"{filtered_df['StockCode'].nunique():,}")

    st.markdown("---")

    # 2. Monthly Trend Chart
    st.subheader("Monthly Sales Volume and Revenue")
    filtered_df["Month"] = filtered_df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    monthly_agg = (
        filtered_df.groupby("Month")
        .agg(
            Recorded_Revenue=("Revenue", "sum"),
            Candidate_Sales=("Candidate_Sales_Revenue", "sum"),
            Units_Sold=("Quantity", lambda s: s[s > 0].sum()),
            Transaction_Count=("Invoice", "count")
        )
        .reset_index()
    )

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=monthly_agg["Month"],
        y=monthly_agg["Units_Sold"],
        name="Units Sold (Standard)",
        yaxis="y2",
        marker_color="rgba(31, 119, 180, 0.3)"
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly_agg["Month"],
        y=monthly_agg["Candidate_Sales"],
        name="Candidate Sales Revenue (£)",
        mode="lines+markers",
        line=dict(color="#2ca02c", width=2.5)
    ))
    fig_trend.update_layout(
        height=380,
        yaxis=dict(title="Revenue (£)", side="left"),
        yaxis2=dict(title="Units Sold", side="right", overlaying="y", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # 3. Top Products and Country Breakdown
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Top 10 Products by Revenue")
        top_prods = (
            filtered_df[filtered_df["Is_Standard_Priced_Sale"]]
            .groupby(["StockCode", "Description"], as_index=False)
            .agg(Total_Revenue=("Revenue", "sum"), Units=("Quantity", "sum"))
            .sort_values("Total_Revenue", ascending=False)
            .head(10)
        )
        fig_prod = px.bar(
            top_prods,
            x="Total_Revenue",
            y="Description",
            orientation="h",
            color="Total_Revenue",
            color_continuous_scale="Viridis",
            labels={"Total_Revenue": "Revenue (£)", "Description": "Product"}
        )
        fig_prod.update_layout(yaxis=dict(autorange="reversed"), height=350, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_prod, use_container_width=True)

    with c2:
        st.subheader("Top 10 Countries by Revenue")
        top_countries = (
            filtered_df[filtered_df["Is_Standard_Priced_Sale"]]
            .groupby("Country", as_index=False)
            .agg(Total_Revenue=("Revenue", "sum"), Invoices=("Invoice", "nunique"))
            .sort_values("Total_Revenue", ascending=False)
            .head(10)
        )
        fig_country = px.bar(
            top_countries,
            x="Country",
            y="Total_Revenue",
            color="Total_Revenue",
            color_continuous_scale="Teal",
            labels={"Total_Revenue": "Revenue (£)"}
        )
        fig_country.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_country, use_container_width=True)

    # 4. Download Filtered Monthly Summary
    st.markdown("### 📥 Download Aggregated Data")
    csv_bytes = monthly_agg.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Monthly Sales Summary (CSV)",
        data=csv_bytes,
        file_name="monthly_sales_summary.csv",
        mime="text/csv"
    )
