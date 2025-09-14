# clustering.py
"""
Run KMeans clustering on the student dataset, choose best k by silhouette,
save model and labelled dataset, and produce plots.

Usage (from project root, venv recommended):
.\venv\Scripts\python.exe clustering.py
"""
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Config ---
DATA_PATH = "student_synthetic.csv"          # change if your dataset filename differs
OUT_CSV = "student_data_with_clusters.csv"
KMEANS_MODEL = "kmeans_model.joblib"
SILHOUETTE_PNG = "kmeans_silhouette.png"
PCA_PNG = "kmeans_clusters_pca.png"
K_RANGE = range(2, 7)  # try k = 2..6

# --- Helpers ---
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)

def preprocess(df):
    # Features used by classifier (and printed earlier)
    features = [
        "hours_sleep", "days_exercised", "study_hours", "social_score",
        "attendance_pct", "gpa", "family_support", "screen_time",
        "concentration_issues", "appetite_change"
    ]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    X = df[features].copy()

    # Encode appetite_change -> numeric (map yes/no or similar)
    # If values are already numeric, this will keep them.
    if X["appetite_change"].dtype == object:
        # common values: "yes"/"no", "Yes"/"No", etc.
        X["appetite_change"] = X["appetite_change"].str.lower().map({"yes": 1, "no": 0})
        # If mapping produced NaN (unknown strings), try label encoding
        if X["appetite_change"].isna().any():
            X["appetite_change"], _ = pd.factorize(df["appetite_change"])
    # Fill remaining missing values (if any) with column median
    X = X.fillna(X.median())

    # Standard scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, X.columns.tolist()

def find_best_k(X, k_range=K_RANGE):
    best_k = None
    best_score = -1
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        results.append((k, score, km, labels))
        print(f"k={k} silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_km = km
            best_labels = labels
    return best_k, best_score, best_km, best_labels, results

def plot_silhouette(X, labels, k, outpath=SILHOUETTE_PNG):
    fig, ax = plt.subplots(figsize=(8, 5))
    sample_silhouette_values = silhouette_samples(X, labels)
    y_lower = 10
    for i in range(k):
        ith_silhouette_vals = sample_silhouette_values[labels == i]
        ith_silhouette_vals.sort()
        size_cluster_i = ith_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_silhouette_vals)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # 10 for spacing
    ax.set_title(f"Silhouette plot for k={k}")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=np.mean(sample_silhouette_values), color="red", linestyle="--")
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print("Saved silhouette plot to", outpath)

def plot_pca_clusters(X, labels, outpath=PCA_PNG):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab10", alpha=0.7)
    ax.set_title("Clusters (PCA projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print("Saved PCA cluster plot to", outpath)

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    X_scaled, scaler, columns = preprocess(df)
    print("Data shape (after preprocessing):", X_scaled.shape)

    print("Searching best k in", list(K_RANGE))
    best_k, best_score, best_km, best_labels, results = find_best_k(X_scaled)
    print(f"Best k: {best_k} (silhouette={best_score:.4f})")

    # Save KMeans model and scaler together as dict
    model_bundle = {
        "kmeans": best_km,
        "scaler": scaler,
        "columns": columns
    }
    joblib.dump(model_bundle, KMEANS_MODEL)
    print("Saved kmeans model to", KMEANS_MODEL)

    # Add cluster labels to original dataframe and save
    df_out = df.copy()
    df_out["cluster"] = best_labels
    df_out.to_csv(OUT_CSV, index=False)
    print("Saved labelled dataset to", OUT_CSV)

    # Plots
    plot_silhouette(X_scaled, best_labels, best_k)
    plot_pca_clusters(X_scaled, best_labels)

if __name__ == "__main__":
    main()
