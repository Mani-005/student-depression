import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# ---------- Helpers: load model(s) ----------
@st.cache_resource
def load_classifier(path="student_depression_model.joblib"):
    if Path(path).exists():
        try:
            clf = joblib.load(path)
            return clf
        except Exception as e:
            st.error(f"Failed to load classifier: {e}")
            return None
    return None

def load_kmeans_and_scaler(path="kmeans_model.joblib"):
    """Return (kmeans_model_or_None, scaler_or_None)."""
    p = Path(path)
    if not p.exists():
        return None, None
    obj = joblib.load(path)
    if hasattr(obj, "predict"):
        return obj, None
    if isinstance(obj, dict):
        kmodel = None
        scaler = None
        if "kmeans" in obj and hasattr(obj["kmeans"], "predict"):
            kmodel = obj["kmeans"]
        if "scaler" in obj:
            scaler = obj["scaler"]
        return kmodel, scaler
    return None, None

# ---------- Analysis UI helper ----------
def show_batch_prediction_analysis(df, top_n=10):
    st.subheader("Prediction analysis — summary")

    total = len(df)
    depressed_count = int((df["pred_label"] == 1).sum())
    not_depressed_count = int((df["pred_label"] == 0).sum())
    depressed_pct = depressed_count / total * 100 if total else 0
    not_depressed_pct = not_depressed_count / total * 100 if total else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total rows", total)
    c2.metric("Predicted Depressed", f"{depressed_count} ({depressed_pct:.1f}%)")
    c3.metric("Predicted Not Depressed", f"{not_depressed_count} ({not_depressed_pct:.1f}%)")

    mean_prob_depr = df.loc[df["pred_label"] == 1, "pred_proba"].mean() if depressed_count else float("nan")
    mean_prob_not = df.loc[df["pred_label"] == 0, "pred_proba"].mean() if not_depressed_count else float("nan")
    st.write(f"Mean predicted probability (depressed): **{mean_prob_depr:.3f}**")
    st.write(f"Mean predicted probability (not depressed): **{mean_prob_not:.3f}**")

    st.subheader("Probability distribution")
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(df["pred_proba"], bins=30, alpha=0.85)
    ax.set_xlabel("Predicted probability (depressed)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader(f"Top {top_n} highest-risk rows")
    top_df = df.sort_values("pred_proba", ascending=False).head(top_n).reset_index(drop=True)
    st.dataframe(top_df)

    buf = io.StringIO()
    top_df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button("Download top risk CSV", buf.getvalue(), file_name="top_risk_students.csv", mime="text/csv")

    st.subheader("Feature means: depressed vs not depressed")
    exclude_cols = {"pred_label", "pred_proba", "cluster"}
    numeric_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        means_by_label = df.groupby("pred_label")[numeric_cols].mean().rename(index={0:"not_depressed",1:"depressed"})
        st.dataframe(means_by_label.T)
        if set([0,1]).issubset(df["pred_label"].unique()):
            diff = means_by_label.loc["depressed"] - means_by_label.loc["not_depressed"]
            st.subheader("Feature differences (depressed - not_depressed)")
            st.bar_chart(diff)
    else:
        st.info("No numeric features available for comparison.")

    if "cluster" in df.columns:
        st.subheader("Cluster breakdown")
        pivot = df.groupby(["cluster", "pred_label"]).size().unstack(fill_value=0)
        pivot = pivot.rename(columns={0:"not_depressed",1:"depressed"})
        st.dataframe(pivot)
        cluster_pct = pivot.div(pivot.sum(axis=1), axis=0).round(3)
        st.write("Cluster-wise percentages:")
        st.dataframe(cluster_pct)

# ---------- Clustering & PCA helper ----------
def assign_clusters_and_pca(df, clustering_model_path="kmeans_model.joblib", scaler_obj=None, n_clusters=2):
    dfc = df.copy()
    exclude = {"pred_label", "pred_proba", "cluster"}
    numeric_cols = [c for c in dfc.columns if c not in exclude and pd.api.types.is_numeric_dtype(dfc[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns available for clustering/PCA.")

    X = dfc[numeric_cols].astype(float).values

    kmeans, k_scaler = load_kmeans_and_scaler(clustering_model_path)
    scaler_to_use = None
    if kmeans is not None:
        scaler_to_use = k_scaler if k_scaler is not None else scaler_obj
        if scaler_to_use is not None:
            X_scaled = scaler_to_use.transform(X)
        else:
            X_scaled = X
        clusters = kmeans.predict(X_scaled)
        used_model = kmeans
    else:
        scaler_to_use = StandardScaler()
        X_scaled = scaler_to_use.fit_transform(X)
        kmeans_tmp = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=64)
        clusters = kmeans_tmp.fit_predict(X_scaled)
        used_model = kmeans_tmp

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    dfc["cluster"] = clusters
    return dfc, X_pca, pca, numeric_cols, used_model, scaler_to_use

# ---------- Main App ----------
st.set_page_config(page_title="Student Depression Prediction + Clustering", layout="wide")
st.title("Student Depression — Predict & Cluster")

model = load_classifier()
if model is None:
    st.error("Classifier not found. Place student_depression_model.joblib in the app folder or run train.py.")
    st.stop()

tab1, tab2 = st.tabs(["🔹 Single Prediction", "📂 Batch Prediction (CSV + Clustering)"])

# --- Tab 1: Single Prediction ---
with tab1:
    st.subheader("Enter features for one student")

    hours_sleep = st.number_input("hours_sleep", 0.0, 12.0, 7.0)
    days_exercised = st.number_input("days_exercised (per week)", 0, 7, 3)
    study_hours = st.number_input("study_hours (per day)", 0.0, 12.0, 3.0)
    social_score = st.number_input("social_score (0-10)", 0, 10, 6)
    attendance_pct = st.number_input("attendance_pct (0-100)", 0, 100, 85)
    gpa = st.number_input("gpa (0-10)", 0.0, 10.0, 7.5)
    family_support = st.selectbox("family_support (0=no,1=yes)", [0,1], index=1)
    screen_time = st.number_input("screen_time (hrs/day)", 0.0, 24.0, 4.0)
    concentration_issues = st.selectbox("concentration_issues (0=no,1=yes)", [0,1], index=0)
    appetite_change = st.selectbox("appetite_change", ["no","yes"], index=0)

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "hours_sleep": hours_sleep,
            "days_exercised": days_exercised,
            "study_hours": study_hours,
            "social_score": social_score,
            "attendance_pct": attendance_pct,
            "gpa": gpa,
            "family_support": family_support,
            "screen_time": screen_time,
            "concentration_issues": concentration_issues,
            "appetite_change": appetite_change
        }])

        try:
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0,1]
        except Exception as e:
            st.error("Prediction failed: " + str(e))
            st.stop()

        label = "Depressed" if pred == 1 else "Not Depressed"
        st.success(f"Prediction: **{label}** (Confidence: {prob:.2f})")

        fig, ax = plt.subplots()
        ax.bar(["Not Depressed","Depressed"], model.predict_proba(input_df)[0], color=["#2ecc71","#e74c3c"])
        ax.set_ylabel("Probability")
        st.pyplot(fig)
        plt.close(fig)

# --- Tab 2: Batch Prediction ---
with tab2:
    st.subheader("Upload a CSV for batch prediction and clustering")
    st.markdown("CSV must include columns: `hours_sleep, days_exercised, study_hours, social_score, attendance_pct, gpa, family_support, screen_time, concentration_issues, appetite_change`")

    uploaded = st.file_uploader("Upload CSV file", type=["csv","xlsx"], key="batch_uploader")
    if uploaded is not None:
        try:
            if str(uploaded.name).lower().endswith(".xlsx") or str(uploaded.name).lower().endswith(".xls"):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        required = ["hours_sleep","days_exercised","study_hours","social_score",
                    "attendance_pct","gpa","family_support","screen_time",
                    "concentration_issues","appetite_change"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        try:
            preds = model.predict(df)
            probs = model.predict_proba(df)[:,1]
            df["pred_label"] = preds
            df["pred_proba"] = probs
        except Exception as e:
            st.error("Failed to predict on uploaded data: " + str(e))
            st.stop()

        st.success("Predictions done. Preview:")
        st.dataframe(df.head(10))

        try:
            df_with_cluster, X_pca, pca_obj, numeric_cols_used, used_kmeans, used_scaler = assign_clusters_and_pca(df, clustering_model_path="kmeans_model.joblib", n_clusters=2)
            st.subheader("Cluster counts")
            st.dataframe(df_with_cluster["cluster"].value_counts().rename_axis("cluster").reset_index(name="count"))

            st.subheader("PCA projection of clusters (uploaded data)")
            fig, ax = plt.subplots(figsize=(6,6))
            clusters_unique = np.unique(df_with_cluster["cluster"])
            cmap = plt.get_cmap("tab10")
            for i, cl in enumerate(sorted(clusters_unique)):
                mask = df_with_cluster["cluster"] == cl
                ax.scatter(X_pca[mask,0], X_pca[mask,1], alpha=0.6, s=25, label=f"Cluster {cl}", color=cmap(i % 10))
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.legend(fontsize="small")
            st.pyplot(fig)
            plt.close(fig)

            df = df_with_cluster

        except Exception as e:
            st.warning("Clustering/PCA could not be completed: " + str(e))

        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button("Download predictions CSV", buf.getvalue(), file_name="predictions_labeled.csv", mime="text/csv")

        show_batch_prediction_analysis(df)
