# 🎓 Student Depression Prediction & Clustering

A machine learning project that predicts **student depression risk** and clusters students into behavioral groups for early intervention.
This project combines **ML pipelines, clustering, and interactive dashboards** into a full-stack solution.

---

## 🚀 Features

* 🔹 **Single Prediction (Flask API + Streamlit)** – Enter one student's data and get depression prediction + confidence score + probability chart.
* 🔹 **Bulk Prediction (CSV Upload)** – Upload multiple student records in Streamlit and get predictions for all.
* 🔹 **Clustering Analysis** – KMeans clustering + PCA visualization to group students by behavior patterns (on bulk data).
* 🔹 **Prediction Analysis Dashboard** – Summarized insights: high-risk students, probability distributions, feature comparison, cluster breakdown.

---

## 🖼️ Workflow Diagram

```
                 ┌────────────────────┐
                 │  make_data.py       │
                 │  (synthetic data)   │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │   train.py          │
                 │ - Preprocessing     │
                 │ - RandomForest      │
                 │ - Save model        │
                 └─────────┬──────────┘
                           │
           ┌───────────────┼────────────────┐
           │                               │
           ▼                               ▼
┌────────────────────┐         ┌────────────────────┐
│   app.py (Flask)   │         │ streamlit_app.py    │
│ - REST API (/predict) │      │ - Single prediction │
│ - JSON input/output │        │ - Bulk prediction   │
└─────────┬──────────┘        │ - Clustering + PCA  │
          │                   │ - Prediction analysis │
          │                   └─────────┬──────────┘
          │                             │
          ▼                             ▼
 API call (JSON)                CSV Upload / UI
 (for devs & apps)              (for teachers/counselors)
```

**Clustering workflow:**

```
student_data.csv → clustering.py → kmeans_model.joblib + student_data_with_clusters.csv
                                     │
                                     ▼
                             streamlit_app.py
                             - Cluster visualization
                             - PCA plots
                             - Bulk cluster assignment
```

---

## 📊 Tech Stack

* **Python 3.10+**
* **Libraries:** Pandas, Numpy, Scikit-learn, Matplotlib
* **Backend:** Flask (REST API)
* **Dashboard:** Streamlit
* **Persistence:** Joblib

---

## ⚙️ Installation & Setup

```bash
# Clone repo
git clone https://github.com/Mani-005/student-depression.git
cd student-depression

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # (Windows)
source venv/bin/activate  # (Linux/Mac)

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Train the Model

```bash
python train.py
```

### 2. Run Flask API (Single Prediction Endpoint)

```bash
python app.py
```

#### Example Test Request (PowerShell):

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST `
  -Body '{"hours_sleep": 7, "days_exercised": 3, "study_hours": 5, "social_score": 6, "attendance_pct": 85, "gpa": 7.5, "family_support": 1, "screen_time": 4, "concentration_issues": 0, "appetite_change": "no"}' `
  -ContentType "application/json"
```

### 3. Run Streamlit Dashboard (Single + Bulk Prediction + Clustering)

```bash
streamlit run streamlit_app.py
```

---

## 📂 Project Structure

```
student-depression/
│── app.py                        # Flask REST API
│── train.py                      # Train RandomForest model
│── clustering.py                  # KMeans clustering + PCA
│── streamlit_app.py               # Streamlit dashboard (Single + Bulk)
│── make_data.py                   # Generate synthetic dataset
│── requirements.txt               # Dependencies
│── student_data.csv               # Dataset
│── student_depression_model.joblib # Saved ML model
│── kmeans_model.joblib            # Saved clustering model
│── README.md                      # Project documentation
```

---

## 🌐 Demo Links

* 🔗 **Streamlit App (Prediction + Clustering)** → [Deployed Link]([https://student-depression-wbxysqc7pu35rottkkvrk7.streamlit.app/](https://student-depression-wbxysqc7pu35rottkkvrk7.streamlit.app/))
* 🔗 **Flask API (Prediction Endpoint)** → Run locally with `python app.py`

---

## 📌 Results

* ✅ **Accuracy:** \~91% (on test set)
* ✅ **ROC AUC:** \~0.98
* 🎯 **Use case:** Helps schools/counselors detect students at risk of depression and group them by behavioral traits.

---

## 👨‍💻 Author

**B.V.S.S.Mani.Teja** –  [GitHub](https://github.com/Mani-005)

---
