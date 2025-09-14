# app.py (debug version)
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import traceback

print("Starting app.py - pid:", os.getpid())

app = Flask(__name__)

MODEL_FILENAME = "student_depression_model.joblib"

# Load model with error handling
model = None
try:
    print("Attempting to load model:", MODEL_FILENAME)
    model = joblib.load(MODEL_FILENAME)
    print("Model loaded successfully. Model type:", type(model))
    # try to access named_steps safely (some pipelines don't have it)
    has_named_steps = hasattr(model, "named_steps")
    print("Model has named_steps?:", has_named_steps)
except Exception as e:
    print("ERROR loading model:", str(e))
    traceback.print_exc()

@app.route("/")
def home():
    return "Flask API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    single = False
    if isinstance(data, dict):
        df = pd.DataFrame([data])
        single = True
    else:
        df = pd.DataFrame(data)

    try:
        # If model supports predict_proba
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[:, 1]
        else:
            # fallback: convert predictions to probabilities (0/1)
            probs = model.predict(df)
        preds = model.predict(df)
    except Exception as e:
        tb = traceback.format_exc()
        print("Model prediction error:", str(e))
        print(tb)
        return jsonify({"error": "Model error: " + str(e), "trace": tb}), 400

    results = []
    for p, pr in zip(preds.tolist(), probs.tolist()):
        results.append({"label": int(p), "probability": float(pr)})

    if single:
        return jsonify(results[0])
    return jsonify(results)

if __name__ == "__main__":
    print("Calling app.run()")
    # Use debug True so you get auto-reload + helpful logs
    app.run(host="0.0.0.0", port=5000, debug=True)
