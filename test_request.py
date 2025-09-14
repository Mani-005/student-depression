import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "hours_sleep": 7,
    "days_exercised": 3,
    "study_hours": 5,
    "social_score": 6,
    "attendance_pct": 85,
    "gpa": 7.5,
    "family_support": 1,
    "screen_time": 4,
    "concentration_issues": 0,
    "appetite_change": "no"  # categorical feature
}

resp = requests.post(url, json=data)
print(resp.json())
