import os
import json
import joblib
from data_loader import load_data
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    df_train, df_test = load_data()
    X_train_p, X_test_p, y_train, y_test, preprocessor = preprocess_data(df_train, df_test)

    model = train_model(X_train_p, y_train)
    evaluate_model(model, X_test_p, y_test)
    
    # Save artifacts
    model_path = os.path.join(MODEL_DIR, "model_v1.pkl")
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    
    print(f"Saving preprocessor to {preprocessor_path}...")
    joblib.dump(preprocessor, preprocessor_path)

    # Save baseline stats for drift detection service
    data_dir = os.path.join(PROJECT_ROOT, "data_storage")
    os.makedirs(data_dir, exist_ok=True)
    numeric_cols = df_train.select_dtypes(include="number").columns.tolist()
    # Exclude metadata columns
    numeric_cols = [c for c in numeric_cols if c not in ["difficulty_level"]]
    baseline = {
        "anomaly_rate": float(y_train.mean()),
        "n_samples": int(len(y_train)),
        "feature_means": {c: float(df_train[c].mean()) for c in numeric_cols},
        "feature_stds":  {c: float(df_train[c].std())  for c in numeric_cols},
    }
    baseline_path = os.path.join(data_dir, "baseline_stats.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Baseline stats saved to {baseline_path}")

    print("ML Pipeline completed successfully.")

if __name__ == "__main__":
    main()
