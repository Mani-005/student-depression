# make_data.py
import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

# realistic-ish distributions
hours_sleep = np.clip(np.random.normal(7, 1.5, n), 2, 12)
days_exercised = np.random.choice(range(0,8), size=n, p=[0.15,0.1,0.15,0.2,0.15,0.1,0.1,0.05])
study_hours = np.clip(np.random.normal(3.5, 2.0, n), 0, 16)
social_score = np.clip(np.random.normal(5, 2.5, n), 0, 10)
attendance_pct = np.clip(np.random.normal(85, 12, n), 30, 100)
gpa = np.clip(np.random.normal(7, 1.5, n), 0, 10)
family_support = np.clip(np.random.normal(6, 2.5, n), 0, 10)
screen_time = np.clip(np.random.normal(5, 2.5, n), 0, 20)
concentration_issues = np.random.binomial(1, 0.25, n)
appetite_change = np.random.choice(["Same", "Increased", "Decreased"], size=n, p=[0.6,0.2,0.2])

# synthetic risk score (toy; for prototyping only)
risk = (
    (7 - hours_sleep) * 0.25
    + (3 - days_exercised) * 0.2
    + (5 - social_score) * 0.2
    + (6 - family_support) * 0.15
    + (screen_time - 3) * 0.05
    + (concentration_issues * 0.5)
    + (np.where(appetite_change=="Decreased", 0.4, 0))
    - (gpa - 6) * 0.1
)

# convert risk to probability and label
prob = 1 / (1 + np.exp(-risk))
label = (prob > np.quantile(prob, 0.7)).astype(int)  # make ~30% labeled depressed

df = pd.DataFrame({
    "hours_sleep": hours_sleep,
    "days_exercised": days_exercised,
    "study_hours": study_hours,
    "social_score": social_score,
    "attendance_pct": attendance_pct,
    "gpa": gpa,
    "family_support": family_support,
    "screen_time": screen_time,
    "concentration_issues": concentration_issues,
    "appetite_change": appetite_change,
    "label": label
})

df.to_csv("student_synthetic.csv", index=False)
print("Saved student_synthetic.csv with", len(df), "rows")
