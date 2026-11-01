"""
Unit tests for inventory calculations.
"""
import pytest
import pandas as pd

from src import config
from src.inventory import calculate_product_inventory_insights, build_inventory_overview_table


def test_inventory_insights_calculation():
    """Verify inventory insights calculation outputs valid reorder point and positive safety stock."""
    df_demand = pd.read_parquet(config.MONTHLY_DEMAND_PARQUET)
    sample_code = df_demand["StockCode"].iloc[0]

    insights = calculate_product_inventory_insights(
        df_demand,
        stock_code=sample_code,
        lead_time_weeks=2.0,
        service_level_z=1.65,
        current_stock_on_hand=100
    )

    assert "estimated_safety_stock" in insights
    assert "suggested_reorder_point" in insights
    assert insights["estimated_safety_stock"] >= 0
    assert insights["suggested_reorder_point"] >= 0
    assert insights["stock_status"] in ["Below Reorder Point", "Potential Overstock", "Healthy Stock Level"]
