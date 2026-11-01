"""
Evaluate demand forecasting models across top products and export benchmark metrics.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config
from src.forecasting import prepare_product_series, evaluate_product_forecasting


def main():
    print("=" * 60)
    print("RETAILPULSE: PRODUCT DEMAND FORECASTING BENCHMARK")
    print("=" * 60)

    # Load monthly demand and product summary
    df_monthly_demand = pd.read_parquet(config.MONTHLY_DEMAND_PARQUET)
    df_product_summary = pd.read_parquet(config.PRODUCT_SUMMARY_PARQUET)

    # Select top 15 eligible products
    eligible_prods = df_product_summary[df_product_summary["Is_Forecasting_Eligible"]].head(15)
    print(f"Evaluating top {len(eligible_prods)} high-history products...")

    benchmarks = []
    for _, row in eligible_prods.iterrows():
        code = row["StockCode"]
        desc = row["Description"]
        try:
            series = prepare_product_series(df_monthly_demand, code)
            res = evaluate_product_forecasting(series, test_periods=3)
            m = res["metrics"]
            benchmarks.append({
                "StockCode": code,
                "Description": desc,
                "Total_Units": int(row["Total_Units_Sold"]),
                "Naive_WAPE": m["Naive"]["WAPE_pct"],
                "Rolling_3M_WAPE": m["Rolling_3M"]["WAPE_pct"],
                "Exp_Smoothing_WAPE": m["Exponential_Smoothing"]["WAPE_pct"],
                "Exp_Smoothing_MAE": m["Exponential_Smoothing"]["MAE"],
                "Exp_Smoothing_RMSE": m["Exponential_Smoothing"]["RMSE"],
                "Best_Model": res["best_model"]
            })
        except Exception as e:
            print(f"Skipping {code}: {e}")

    df_bench = pd.DataFrame(benchmarks)
    print("\n--- Summary Benchmark Across Top Products ---")
    print(f"Mean Naive WAPE: {df_bench['Naive_WAPE'].mean():.2f}%")
    print(f"Mean Rolling 3M WAPE: {df_bench['Rolling_3M_WAPE'].mean():.2f}%")
    print(f"Mean Exp Smoothing WAPE: {df_bench['Exp_Smoothing_WAPE'].mean():.2f}%")

    # Export metrics
    config.FORECASTING_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FORECASTING_MODEL_DIR / "forecast_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(df_bench.to_dict(orient="records"), f, indent=2)

    print(f"\n[DONE] Forecasting evaluation metrics exported to {out_path}")


if __name__ == "__main__":
    main()
