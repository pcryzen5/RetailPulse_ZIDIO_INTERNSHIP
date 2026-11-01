"""
Execution script to generate all processed Parquet datasets and
export customer segmentation model artifacts.
"""
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_loader import process_and_save_all
from src.segmentation import train_and_export_segmentation_model


def main():
    print("=" * 60)
    print("RETAILPULSE: DATA PROCESSING & MODEL ARTIFACT GENERATOR")
    print("=" * 60)
    start_time = time.time()

    # 1. End-to-end data processing and Parquet persistence
    datasets = process_and_save_all()
    df_rfm = datasets["rfm_customers"]

    print("\n--- Summary of Generated Datasets ---")
    for name, df in datasets.items():
        print(f"  {name:22}: {len(df):,} rows | {len(df.columns)} columns")

    # 2. Customer segmentation training and serialization
    print("\n--- Training & Serializing Customer Segmentation Model ---")
    kmeans, scaler, rfm_clustered, meta = train_and_export_segmentation_model(df_rfm, export=True)

    print(f"Total customers evaluated: {meta['total_customers']:,}")
    print(f"K-Means silhouette score : {meta['silhouette_score']}")
    print(f"PCA explained variance   : {meta['pca_explained_variance_pct']}%")
    print("\nCluster Distribution:")
    for cluster, count in sorted(meta["cluster_distribution"].items()):
        seg = meta["label_mapping"].get(cluster, "Unknown")
        print(f"  Cluster {cluster} ({seg}): {count:,} customers ({count/meta['total_customers']*100:.1f}%)")

    elapsed = time.time() - start_time
    print(f"\n[DONE] Pipeline completed successfully in {elapsed:.1f} seconds!")


if __name__ == "__main__":
    main()
