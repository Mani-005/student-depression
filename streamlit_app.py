# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from io import BytesIO


st.set_page_config(page_title="Student Depression ", layout="centered")

# Filenames (adjust if you used different names)
MODEL_FILE = "student_depression_model.joblib"
KMEANS_FILE = "kmeans_model.joblib"
CLUSTERED_CSV = "student_data_with_clusters.csv"
PCA_IMG = "kmeans_clusters_pca.png"

# Helper to load model safely
@st.cache_resource
def load_classifier(path=MODEL_FILE):
    p = Path(path)
    if not p.exists():
        return None, f"Classifier file not found: {path}"
    try:
        m = joblib.load(path)
        return m, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_kmeans_bundle(path=KMEANS_FILE):
    p = Path(path)
    if not p.exists():
        return None, f"KMeans bundle not found: {path}"
    try:
        bundle = joblib.load(path)
        return bundle, None
    except Exception as e:
        return None, str(e)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home / Predict", "Clustering"])

# ---------- Home / Predict ----------
# ---------- Home / Predict (Batch-only version) ----------
if page == "Home / Predict":
    st.title("Student Depression — Batch Predict (CSV)")

    clf, err = load_classifier()
    if err:
        st.error(f"Model load error: {err}")
        st.info("Run python train.py first to create the classifier model file.")
        st.stop()

    st.markdown(
        "Upload a CSV (one row per student) to get batch predictions. "
        "CSV must contain these columns: "
        "hours_sleep, days_exercised, study_hours, social_score, attendance_pct, gpa, family_support, screen_time, concentration_issues, appetite_change."
    )

    uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            needed = [
                "hours_sleep","days_exercised","study_hours","social_score",
                "attendance_pct","gpa","family_support","screen_time",
                "concentration_issues","appetite_change"
            ]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            else:
                preds = clf.predict(df)
                probs = clf.predict_proba(df)[:,1] if hasattr(clf, "predict_proba") else preds
                out = df.copy()
                out["pred_label"] = preds
                out["pred_proba"] = probs
                st.success("Predictions done. Preview:")
                st.dataframe(out.head(10))
                buf = BytesIO()
                out.to_csv(buf, index=False)
                buf.seek(0)
                st.download_button("Download predictions CSV", data=buf, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error("Failed to run batch prediction: " + str(e))

    st.title("Student Depression — Predict ")
    clf, err = load_classifier()
    if err:
        st.error(f"Model load error: {err}")
        st.info("Run python train.py first to create the classifier model file.")
        st.stop()

    st.markdown("Enter feature values for a single student and press Predict.")
    # Feature inputs (match the model's features)
    col1, col2 = st.columns(2)
    with col1:
        hours_sleep = st.number_input("hours_sleep", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        days_exercised = st.number_input("days_exercised (per week)", min_value=0, max_value=7, value=3)
        study_hours = st.number_input("study_hours (per day)", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
        social_score = st.number_input("social_score (0-10)", min_value=0, max_value=10, value=6)
        attendance_pct = st.number_input("attendance_pct (0-100)", min_value=0, max_value=100, value=85)
    with col2:
        gpa = st.number_input("gpa (0-10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        family_support = st.selectbox("family_support (0=no, 1=yes)", options=[0,1], index=1)
        screen_time = st.number_input("screen_time (hrs/day)", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
        concentration_issues = st.selectbox("concentration_issues (0=no,1=yes)", options=[0,1], index=0)
        appetite_change = st.selectbox("appetite_change", options=["no","yes"], index=0)

    if st.button("Predict"):
    # Build dataframe expected by your model pipeline
     x = pd.DataFrame([{
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
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(x)[:, 1][0]
        else:
            proba = float(clf.predict(x)[0])

        pred = int(clf.predict(x)[0])
        status = "Depressed" if pred == 1 else "Not Depressed"

        st.success(f"Prediction: *{status}* (Confidence: {proba:.2f})")
    except Exception as e:
        st.error("Model error: " + str(e))

   
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            # basic check
            needed = ["hours_sleep","days_exercised","study_hours","social_score","attendance_pct","gpa","family_support","screen_time","concentration_issues","appetite_change"]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            else:
                preds = clf.predict(df)
                probs = clf.predict_proba(df)[:,1] if hasattr(clf, "predict_proba") else preds
                out = df.copy()
                out["pred_label"] = preds
                out["pred_proba"] = probs
                st.success("Predictions done. Preview:")
                st.dataframe(out.head(10))
                buf = BytesIO()
                out.to_csv(buf, index=False)
                buf.seek(0)
                st.download_button("Download predictions CSV", data=buf, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error("Failed to run batch prediction: " + str(e))

# ---------- Clustering ----------
else:
    st.title("Clustering analysis")
    bundle, err = load_kmeans_bundle()
    if err:
        st.error(f"KMeans bundle error: {err}")
        st.info("Run python clustering.py first to create clustering outputs.")
        st.stop()

    # Load labelled CSV if exists
    csv_path = Path(CLUSTERED_CSV)
    if not csv_path.exists():
        st.error(f"Clustered CSV not found: {CLUSTERED_CSV}. Run clustering.py")
        st.stop()

    dfc = pd.read_csv(csv_path)
    st.subheader("Cluster counts")
    st.write(dfc["cluster"].value_counts().sort_index())

    st.subheader("Cluster summary (means)")
    st.write(dfc.groupby("cluster").mean(numeric_only=True))


    st.subheader("Browse cluster samples")
    sel = st.selectbox("Choose cluster", sorted(dfc["cluster"].unique()))
    st.dataframe(dfc[dfc["cluster"] == sel].reset_index(drop=True).head(30))

    # show pca image if generated
    pimg = Path(PCA_IMG)
    if pimg.exists():
        st.subheader("PCA projection of clusters")
        st.image(str(pimg), use_container_width=True)

    # download labeled CSV
    buf = BytesIO()
    dfc.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button("Download labeled CSV", data=buf, file_name="student_data_with_clusters.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("If you'd like, upload a CSV to assign cluster IDs to new rows (using existing KMeans):")
    up = st.file_uploader("Upload CSV to assign clusters", type=["csv"])
    if up is not None:
        try:
            new_df = pd.read_csv(up)
            # basic preprocessing: map appetite_change strings to 0/1 if needed
            if "appetite_change" in new_df.columns and new_df["appetite_change"].dtype == object:
                new_df["appetite_change"] = new_df["appetite_change"].str.lower().map({"yes":1,"no":0})
                if new_df["appetite_change"].isna().any():
                    new_df["appetite_change"], _ = pd.factorize(new_df["appetite_change"])
            # use bundle scaler + kmeans
            kmeans = bundle["kmeans"]
            scaler = bundle["scaler"]
            cols = bundle["columns"]
            missing = [c for c in cols if c not in new_df.columns]
            if missing:
                st.error("Missing columns in uploaded file: " + ", ".join(missing))
            else:
                X = new_df[cols].fillna(new_df[cols].median())
                Xs = scaler.transform(X)
                labels = kmeans.predict(Xs)
                new_df["cluster"] = labels
                st.success("Assigned clusters. Preview:")
                st.dataframe(new_df.head(20))
                buf2 = BytesIO()
                new_df.to_csv(buf2, index=False)
                buf2.seek(0)
                st.download_button("Download with clusters", data=buf2, file_name="uploaded_with_clusters.csv", mime="text/csv")
        except Exception as e:
            st.error("Error processing uploaded file: " + str(e))

# Footer
