"""
retrain.py
----------
Drift-triggered retraining script.

Called by the drift service (drift_service/app.py) when drift is detected.
Unlike main.py (which uses the 70% initial split), this script trains on
the FULL dataset (initial 70% + reserve 30%) so the model improves
by seeing data it was never exposed to during initial training.

Flow:
  drift detected → drift_service calls retrain.py → new .pkl files saved
                → drift_service calls POST /reload on prediction_service
                → prediction service hot-swaps model with no restart
"""

import os
import json
import joblib
from data_loader import load_retrain_data
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_storage")


def retrain():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 55)
    print("  DRIFT-TRIGGERED RETRAINING")
    print("  Dataset: initial (70%) + reserve (30%) combined")
    print("=" * 55)

    # Use combined dataset (initial + reserve)
    df_train, df_test = load_retrain_data()

    X_train_p, X_test_p, y_train, y_test, preprocessor = preprocess_data(df_train, df_test)

    model = train_model(X_train_p, y_train)
    evaluate_model(model, X_test_p, y_test)

    # Overwrite existing model artifacts
    model_path       = os.path.join(MODEL_DIR, "model_v1.pkl")
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")

    print(f"Saving retrained model to {model_path}...")
    joblib.dump(model, model_path)

    print(f"Saving preprocessor to {preprocessor_path}...")
    joblib.dump(preprocessor, preprocessor_path)

    # Update baseline stats so the next drift check compares against the new model
    os.makedirs(DATA_DIR, exist_ok=True)
    numeric_cols = df_train.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["difficulty_level"]]
    baseline = {
        "anomaly_rate": float(y_train.mean()),
        "n_samples": int(len(y_train)),
        "feature_means": {c: float(df_train[c].mean()) for c in numeric_cols},
        "feature_stds":  {c: float(df_train[c].std())  for c in numeric_cols},
    }
    baseline_path = os.path.join(DATA_DIR, "baseline_stats.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Baseline stats updated at {baseline_path}")

    print("Retraining completed successfully.")


if __name__ == "__main__":
    retrain()
