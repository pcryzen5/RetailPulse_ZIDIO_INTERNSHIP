"""
Product Demand Forecasting View for RetailPulse Dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any

from src.forecasting import (
    prepare_product_series,
    evaluate_product_forecasting,
    generate_future_forecast
)


def render_demand_forecasting(df_monthly_demand: pd.DataFrame, df_product_summary: pd.DataFrame):
    st.title("📦 Product-Level Demand Forecasting")
    st.markdown(
        "Forecast monthly demand using **Exponential Smoothing** rigorously benchmarked against "
        "**Naive** and **Rolling Average baselines** using held-out chronological evaluation."
    )

    if df_monthly_demand.empty:
        st.warning("Monthly demand dataset not found.")
        return

    # Product selector
    eligible_prods = df_product_summary[df_product_summary["Is_Forecasting_Eligible"]].copy()
    if eligible_prods.empty:
        eligible_prods = df_product_summary.head(20).copy()

    options = {
        f"{row['StockCode']} - {row['Description'][:35]} ({row['Total_Units_Sold']:,} units)": row['StockCode']
        for _, row in eligible_prods.iterrows()
    }

    sc1, sc2 = st.columns([3, 1])
    with sc1:
        selected_label = st.selectbox("Select Product to Forecast", options=list(options.keys()), index=0)
        selected_code = options[selected_label]
    with sc2:
        forecast_horizon = st.slider("Forecast Horizon (Months)", min_value=1, max_value=6, value=3)

    # Prepare time-series
    try:
        series = prepare_product_series(df_monthly_demand, selected_code)
    except Exception as e:
        st.error(f"Error loading series for {selected_code}: {e}")
        return

    # 1. Model Evaluation on Held-Out Test Set (Last 3 Months)
    test_periods = min(3, len(series) // 3)
    if test_periods >= 1 and len(series) >= 6:
        eval_results = evaluate_product_forecasting(series, test_periods=test_periods)
        metrics = eval_results["metrics"]

        st.subheader("🎯 Chronological Held-Out Evaluation (Last 3 Months)")
        st.caption("No shuffling, no data leakage. Models evaluated on held-out test months.")

        m1, m2, m3, m4 = st.columns(4)
        best_model = eval_results["best_model"]
        ets_m = metrics["Exponential_Smoothing"]
        rolling_m = metrics["Rolling_3M"]
        naive_m = metrics["Naive"]

        with m1:
            st.metric("Exp Smoothing WAPE", f"{ets_m['WAPE_pct']:.1f}%", f"Rolling: {rolling_m['WAPE_pct']:.1f}%", delta_color="inverse")
        with m2:
            st.metric("Exp Smoothing MAE", f"{ets_m['MAE']:.1f} units", f"Naive: {naive_m['MAE']:.1f}", delta_color="inverse")
        with m3:
            st.metric("Exp Smoothing RMSE", f"{ets_m['RMSE']:.1f}")
        with m4:
            st.metric("Selected Champion", best_model.replace("_", " "))

    # 2. Forward-Looking Horizon Forecast
    future_forecast = generate_future_forecast(series, horizon=forecast_horizon, model_type="Exponential_Smoothing")

    st.subheader(f"📈 Forward Forecast: Next {forecast_horizon} Months")

    fig = go.Figure()
    # Historical Actuals
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines+markers",
        name="Historical Actuals",
        line=dict(color="#1f77b4", width=2.5)
    ))

    # Future Point Forecast
    fig.add_trace(go.Scatter(
        x=future_forecast["Month"],
        y=future_forecast["Forecast_Units"],
        mode="lines+markers",
        name="Exponential Smoothing Forecast",
        line=dict(color="#2ca02c", width=3, dash="dash")
    ))

    # Upper and Lower Confidence Bounds
    fig.add_trace(go.Scatter(
        x=list(future_forecast["Month"]) + list(future_forecast["Month"])[::-1],
        y=list(future_forecast["Upper_CI_95"]) + list(future_forecast["Lower_CI_95"])[::-1],
        fill="toself",
        fillcolor="rgba(44, 160, 44, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% Uncertainty Bound"
    ))

    fig.update_layout(
        height=420,
        xaxis_title="Month",
        yaxis_title="Units Demanded",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. Forecast Table & Baseline Comparison
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Projected Monthly Demand Numbers")
        st.dataframe(future_forecast[["Month", "Forecast_Units", "Lower_CI_95", "Upper_CI_95"]], use_container_width=True)

    with c2:
        st.markdown("#### Baseline Comparison Summary")
        if 'eval_results' in locals():
            comp_df = pd.DataFrame([
                {"Model": "Naive Baseline", "MAE": naive_m["MAE"], "RMSE": naive_m["RMSE"], "WAPE (%)": naive_m["WAPE_pct"]},
                {"Model": "Rolling 3-Month Average", "MAE": rolling_m["MAE"], "RMSE": rolling_m["RMSE"], "WAPE (%)": rolling_m["WAPE_pct"]},
                {"Model": "Exponential Smoothing", "MAE": ets_m["MAE"], "RMSE": ets_m["RMSE"], "WAPE (%)": ets_m["WAPE_pct"]}
            ])
            st.dataframe(comp_df, use_container_width=True)

    # 4. Known Forecasting Limitations
    st.markdown("---")
    st.info(
        "⚠️ **Forecasting Scope & Limitations:**\n"
        "- Historical data spans 25 calendar months (Dec 2009 - Dec 2011). Partial monthly records exist in Dec 2009 and Dec 2011.\n"
        "- Product demand histories can be intermittent; months with no recorded sales are explicitly modeled as 0 demand.\n"
        "- Statistical Exponential Smoothing is chosen for robust generalization on short retail series over complex neural approaches."
    )
