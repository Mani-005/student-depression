# print_model_inputs.py
import joblib, traceback
MODEL = "student_depression_model.joblib"

try:
    m = joblib.load(MODEL)
    print("Model type:", type(m))
    # 1) try feature_names_in_ (common for estimators)
    if hasattr(m, "feature_names_in_"):
        print("feature_names_in_ found:")
        for c in m.feature_names_in_:
            print("  -", c)
    # 2) try pipeline preprocessor ColumnTransformer (common pattern)
    try:
        pp = m.named_steps.get("preprocessor", None) if hasattr(m, "named_steps") else None
        if pp is not None:
            print("\nColumnTransformer transformers_ info:")
            for transformer in pp.transformers_:
                name, trans, cols = transformer
                print(f"Transformer: {name} | columns: {cols}")
    except Exception as e:
        print("Could not inspect preprocessor:", e)
    # 3) try to show any training columns saved in the model (custom)
    if hasattr(m, "get_params"):
        # some models store columns in steps or attributes; try common names:
        for attr in ("columns", "feature_names", "input_columns", "feature_names_in_"):
            if hasattr(m, attr):
                print(f"\nFound attribute {attr}:")
                print(getattr(m, attr))
except Exception:
    print("Failed to load/inspect model:")
    traceback.print_exc()
