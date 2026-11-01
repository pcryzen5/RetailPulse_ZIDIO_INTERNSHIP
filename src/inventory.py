"""
RetailPulse Demand-Based Inventory Insights Module.
Translates historical demand volatility and forecast projections into operational
inventory planning estimates under explicit, user-configurable assumptions.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import math


def calculate_product_inventory_insights(
    df_monthly_demand: pd.DataFrame,
    stock_code: str,
    lead_time_weeks: float = 2.0,
    service_level_z: float = 1.65,  # ~95% service level
    forecast_units: Optional[float] = None,
    current_stock_on_hand: Optional[float] = None
) -> Dict[str, Any]:
    """
    Generate demand-driven inventory guidance for a single product under explicit assumptions.
    
    Caveats:
    - Source dataset does NOT contain actual warehouse stock-on-hand or supplier lead times.
    - All recommendations are demand-based statistical estimates, not verified ground truth.
    """
    prod_demand = df_monthly_demand[df_monthly_demand["StockCode"] == str(stock_code)].copy()
    if prod_demand.empty:
        raise ValueError(f"No demand data available for StockCode {stock_code}")

    prod_demand = prod_demand.sort_values("Month")
    units = prod_demand["Units_Sold"].values
    desc = prod_demand["Description"].iloc[0] if "Description" in prod_demand.columns else "Product"

    # Historical demand statistics
    recent_3m_avg = float(np.mean(units[-3:])) if len(units) >= 3 else float(np.mean(units))
    recent_std = float(np.std(units[-3:])) if len(units) >= 3 else (float(np.std(units)) if len(units) > 1 else 0.0)
    all_time_avg = float(np.mean(units))

    # Trend detection
    if len(units) >= 6:
        prev_3m_avg = float(np.mean(units[-6:-3]))
        if recent_3m_avg > prev_3m_avg * 1.15:
            trend = "Increasing Demand"
        elif recent_3m_avg < prev_3m_avg * 0.85:
            trend = "Decreasing Demand"
        else:
            trend = "Stable Demand"
    else:
        trend = "Limited History"

    # Weekly demand rate (approx 4.33 weeks per month)
    weekly_demand_rate = recent_3m_avg / 4.33
    weekly_std = recent_std / math.sqrt(4.33) if recent_std > 0 else 1.0

    # Lead time demand & safety stock estimates
    lead_time_demand = weekly_demand_rate * lead_time_weeks
    safety_stock = service_level_z * weekly_std * math.sqrt(max(lead_time_weeks, 0.1))
    reorder_point = lead_time_demand + safety_stock

    # Forecast-informed monthly requirement
    planning_base_demand = forecast_units if forecast_units is not None else recent_3m_avg
    suggested_order_qty = planning_base_demand + safety_stock

    stock_status = "Not Provided"
    action_recommendation = "Maintain regular review schedule."
    if current_stock_on_hand is not None:
        if current_stock_on_hand <= reorder_point:
            stock_status = "Below Reorder Point"
            action_recommendation = f"Consider placing reorder of ~{int(round(suggested_order_qty - current_stock_on_hand))} units."
        elif current_stock_on_hand > suggested_order_qty * 1.5:
            stock_status = "Potential Overstock"
            action_recommendation = "Hold further orders until inventory normalizes."
        else:
            stock_status = "Healthy Stock Level"
            action_recommendation = "Stock is within calculated buffer levels."

    return {
        "stock_code": str(stock_code),
        "description": desc,
        "recent_3m_avg_monthly_units": round(recent_3m_avg, 1),
        "recent_std_monthly_units": round(recent_std, 1),
        "all_time_avg_monthly_units": round(all_time_avg, 1),
        "demand_trend": trend,
        "assumptions": {
            "lead_time_weeks": lead_time_weeks,
            "service_level_z": service_level_z,
            "weeks_per_month": 4.33
        },
        "estimated_lead_time_demand": round(lead_time_demand, 1),
        "estimated_safety_stock": round(safety_stock, 1),
        "suggested_reorder_point": round(reorder_point, 1),
        "suggested_buffer_stock": round(suggested_order_qty, 1),
        "current_stock_on_hand": current_stock_on_hand,
        "stock_status": stock_status,
        "action_recommendation": action_recommendation,
        "disclaimer": (
            "Notice: These calculations are statistical estimates derived from historical demand. "
            "Actual supplier lead times, storage capacities, and stock-on-hand were not recorded in the dataset."
        )
    }


def build_inventory_overview_table(
    df_monthly_demand: pd.DataFrame,
    lead_time_weeks: float = 2.0,
    service_level_z: float = 1.65,
    top_n: int = 50
) -> pd.DataFrame:
    """
    Build a multi-product inventory summary table for the top active products.
    """
    top_products = (
        df_monthly_demand.groupby("StockCode")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    records = []
    for code in top_products:
        try:
            item = calculate_product_inventory_insights(
                df_monthly_demand,
                stock_code=code,
                lead_time_weeks=lead_time_weeks,
                service_level_z=service_level_z
            )
            records.append({
                "StockCode": item["stock_code"],
                "Description": item["description"],
                "Recent_3M_Avg_Units": item["recent_3m_avg_monthly_units"],
                "Demand_Trend": item["demand_trend"],
                "Lead_Time_Demand": item["estimated_lead_time_demand"],
                "Safety_Stock": item["estimated_safety_stock"],
                "Suggested_Reorder_Point": item["suggested_reorder_point"]
            })
        except Exception:
            continue

    return pd.DataFrame(records)
