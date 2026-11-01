"""
Train and export churn classification models and metrics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config
from src.churn import build_churn_dataset, train_and_evaluate_churn_models


def main():
    print("=" * 60)
    print("RETAILPULSE: CHURN CLASSIFIER TRAINING")
    print("=" * 60)

    # Load cleaned transactions
    df_cleaned = pd.read_parquet(config.CLEANED_TRANSACTIONS_PARQUET)

    print("Building churn dataset with leakage-free 90-day forward window...")
    df_churn, meta = build_churn_dataset(df_cleaned)

    print(f"Eligible customers: {meta['eligible_customers']:,}")
    print(f"Churn rate: {meta['churn_rate_pct']}% ({meta['churn_count'] if 'churn_count' in meta else meta['churned_count']} churned, {meta['active_count']} active)")

    print("\nTraining Logistic Regression baseline and Random Forest...")
    results = train_and_evaluate_churn_models(df_churn, export=True)

    lr_m = results["baseline_logistic_regression"]
    rf_m = results["random_forest"]

    print("\n--- Model Evaluation Results ---")
    print(f"Logistic Regression: ROC-AUC={lr_m['ROC_AUC']}, PR-AUC={lr_m['PR_AUC']}, F1={lr_m['F1_Score']}")
    print(f"Random Forest      : ROC-AUC={rf_m['ROC_AUC']}, PR-AUC={rf_m['PR_AUC']}, F1={rf_m['F1_Score']}")
    print("\nRandom Forest Feature Importances:")
    for feat, imp in sorted(results["feature_importances"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:18}: {imp:.4f}")

    print("\n[DONE] Churn artifacts successfully generated and saved!")


if __name__ == "__main__":
    main()
