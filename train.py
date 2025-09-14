# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
from sklearn.preprocessing import OneHotEncoder
import inspect

# Check which argument is supported
if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

cat_pipeline = Pipeline([("onehot", onehot)])


df = pd.read_csv("student_synthetic.csv")

# features
features = [c for c in df.columns if c not in ("label", "id")]

# numeric vs categorical
numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in features if c not in numeric_features]

X = df[features]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

num_pipeline = Pipeline([("scaler", StandardScaler())])
cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])


preprocessor = ColumnTransformer([
    ("num", num_pipeline, numeric_features),
    ("cat", cat_pipeline, categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [5, 10, None]
}

gs = GridSearchCV(pipeline, param_grid, cv=4, scoring="roc_auc", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

print("Best params:", gs.best_params_)
best = gs.best_estimator_

y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(best, "student_depression_model.joblib")
print("Saved student_depression_model.joblib")
