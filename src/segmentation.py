"""
RetailPulse Customer Segmentation Module.
Implements RFM log transformation, standardization, K-Means clustering (K=4),
cluster profiling, PCA projections for visualization, and model serialization.
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src import config


def train_and_export_segmentation_model(
    df_rfm: pd.DataFrame,
    export: bool = True
) -> Tuple[KMeans, StandardScaler, pd.DataFrame, Dict[str, Any]]:
    """
    Train and export the standard RetailPulse K-Means (K=4) clustering model.
    Uses log1p transformation followed by StandardScaler, exactly matching
    the validated research notebook.
    """
    rfm = df_rfm.copy()
    features = config.CLUSTERING_FEATURES

    # 1. Log1p transformation to mitigate right-skewness
    rfm_log = np.log1p(rfm[features])

    # 2. StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm_log)

    # 3. K-Means clustering with K=4, random_state=42, n_init=10
    kmeans = KMeans(
        n_clusters=config.KMEANS_CLUSTERS,
        random_state=config.RANDOM_STATE,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(scaled_features)
    rfm["Cluster"] = cluster_labels

    # 4. Two-component PCA for visualization
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    pca_coords = pca.fit_transform(scaled_features)
    rfm["PCA_Dim1"] = pca_coords[:, 0]
    rfm["PCA_Dim2"] = pca_coords[:, 1]
    explained_var = float(np.sum(pca.explained_variance_ratio_) * 100)

    # 5. Evaluate silhouette score
    sil_score = float(silhouette_score(scaled_features, cluster_labels))

    # 6. Compute empirical cluster profiles to assign provisional business descriptions
    profile = (
        rfm.groupby("Cluster")
        .agg(
            Customer_Count=("Customer ID", "count"),
            Median_Recency=("Recency", "median"),
            Median_Frequency=("Frequency", "median"),
            Median_Monetary=("Monetary", "median"),
            Mean_Recency=("Recency", "mean"),
            Mean_Frequency=("Frequency", "mean"),
            Mean_Monetary=("Monetary", "mean"),
        )
        .reset_index()
    )

    # Map provisional business segment names based on empirical statistics
    # Lowest recency + highest frequency/monetary = High-Value Champions
    # High recency + low frequency/monetary = Dormant / Lost
    # High frequency/monetary + moderate/high recency = At Risk
    def get_provisional_label(row):
        r_med = row["Median_Recency"]
        f_med = row["Median_Frequency"]
        m_med = row["Median_Monetary"]
        if r_med < 30 and f_med >= 6:
            return "High-Value Champions"
        elif r_med > 180 and f_med <= 2:
            return "Dormant / Lost"
        elif r_med > 60 and (f_med >= 4 or m_med > 1500):
            return "At Risk - High Value"
        else:
            return "Occasional / Regular"

    profile["Provisional_Segment"] = profile.apply(get_provisional_label, axis=1)
    label_map = dict(zip(profile["Cluster"], profile["Provisional_Segment"]))
    rfm["Segment_Name"] = rfm["Cluster"].map(label_map)

    # Model metadata
    metadata = {
        "n_clusters": config.KMEANS_CLUSTERS,
        "silhouette_score": round(sil_score, 4),
        "inertia": round(float(kmeans.inertia_), 2),
        "pca_explained_variance_pct": round(explained_var, 2),
        "total_customers": len(rfm),
        "cluster_distribution": rfm["Cluster"].value_counts().to_dict(),
        "cluster_profiles": profile.to_dict(orient="records"),
        "label_mapping": label_map
    }

    # 7. Serialization
    if export:
        config.SEGMENTATION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, config.RFM_SCALER_PATH)
        joblib.dump(kmeans, config.KMEANS_MODEL_PATH)
        with open(config.SEGMENT_PROFILES_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Update saved parquet with cluster labels and PCA coordinates
        rfm.to_parquet(config.RFM_CUSTOMERS_PARQUET, index=False)
        print(f"[Segmentation] Saved scaler, KMeans model, and profiles to {config.SEGMENTATION_MODEL_DIR}")

    return kmeans, scaler, rfm, metadata


def load_segmentation_artifacts() -> Tuple[KMeans, StandardScaler, Dict[str, Any]]:
    """
    Load pre-trained KMeans model, StandardScaler, and profiles metadata.
    """
    if not config.KMEANS_MODEL_PATH.exists() or not config.RFM_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Segmentation artifacts missing in {config.SEGMENTATION_MODEL_DIR}. Run training first."
        )

    kmeans = joblib.load(config.KMEANS_MODEL_PATH)
    scaler = joblib.load(config.RFM_SCALER_PATH)

    metadata = {}
    if config.SEGMENT_PROFILES_PATH.exists():
        with open(config.SEGMENT_PROFILES_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return kmeans, scaler, metadata


def predict_customer_segment(
    recency: float,
    frequency: float,
    monetary: float,
    kmeans: Optional[KMeans] = None,
    scaler: Optional[StandardScaler] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Predict cluster and provisional business segment for a single customer
    using the pre-fitted preprocessing scaler and KMeans model.
    """
    if kmeans is None or scaler is None:
        kmeans, scaler, meta = load_segmentation_artifacts()
    else:
        meta = metadata or {}

    raw_input = np.array([[recency, frequency, monetary]], dtype=float)
    log_input = np.log1p(np.maximum(raw_input, 0.0))
    scaled_input = scaler.transform(log_input)

    cluster_id = int(kmeans.predict(scaled_input)[0])
    label_map = meta.get("label_mapping", {})
    segment_name = label_map.get(str(cluster_id), label_map.get(cluster_id, f"Cluster {cluster_id}"))

    return {
        "cluster_id": cluster_id,
        "segment_name": segment_name,
        "scaled_features": scaled_input[0].tolist()
    }
