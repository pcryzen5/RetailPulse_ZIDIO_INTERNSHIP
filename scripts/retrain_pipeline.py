"""
RetailPulse Automated Retraining Pipeline.
Triggered automatically when data drift exceeds tolerance (PSI >= 0.25)
or scheduled as a periodic batch pipeline.
"""
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config
from src.data_loader import process_and_save_all
from src.segmentation import train_and_export_segmentation_model
from src.churn import build_churn_dataset, train_and_evaluate_churn_models
from src.forecasting import prepare_product_series, evaluate_product_forecasting


def run_automated_retraining():
    print("=" * 70)
    print("RETAILPULSE: MLOps AUTOMATED RETRAINING PIPELINE")
    print("=" * 70)
    start_time = time.time()

    # 1. Pipeline trigger audit
    print("[MLOps Trigger] Retraining initiated due to scheduled cron or data drift trigger.")

    # 2. Re-process datasets
    print("[Step 1/3] Refreshing processed datasets from transactional store...")
    datasets = process_and_save_all()
    df_rfm = datasets["rfm_customers"]
    df_cleaned = datasets["cleaned_transactions"]

    # 3. Retrain Customer Segmentation (K-Means K=4)
    print("\n[Step 2/3] Retraining & updating Customer Segmentation model...")
    km, scaler, rfm_clustered, meta_seg = train_and_export_segmentation_model(df_rfm, export=True)
    print(f"  Segmentation Model Updated. Silhouette Score: {meta_seg['silhouette_score']}")

    # 4. Retrain Customer Churn Classifier
    print("\n[Step 3/3] Retraining & updating Churn Classification model...")
    df_churn, meta_churn = build_churn_dataset(df_cleaned)
    churn_results = train_and_evaluate_churn_models(df_churn, export=True)
    rf_metrics = churn_results["random_forest"]
    print(f"  Churn Classifier Updated. Random Forest ROC-AUC: {rf_metrics['ROC_AUC']}, F1: {rf_metrics['F1_Score']}")

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Automated MLOps retraining pipeline completed in {elapsed:.1f}s.")
    print("Production artifacts refreshed in models/ directory.")


if __name__ == "__main__":
    run_automated_retraining()
