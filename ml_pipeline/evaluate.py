import numpy as np
from sklearn.metrics import classification_report, accuracy_score

LABEL_MAP = {0: "Normal", 1: "Anomaly (Known)", 2: "Anomaly (Unknown/Zero-Day)"}

def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    # For the classification report, collapse 1 and 2 both into "Anomaly"
    # so we can compare against the binary y_test ground truth (0=normal, 1=anomaly)
    y_pred_binary = np.where(y_pred > 0, 1, 0)

    print("Classification Report (Normal vs Any Anomaly):")
    print(classification_report(y_test, y_pred_binary, target_names=["Normal", "Anomaly"]))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_binary):.4f}")

    # Also show the breakdown by detection type
    known   = np.sum(y_pred == 1)
    unknown = np.sum(y_pred == 2)
    normal  = np.sum(y_pred == 0)
    print(f"\nHybrid Model Breakdown:")
    print(f"  Normal              : {normal}")
    print(f"  Anomaly (Known)     : {known}   ← caught by Random Forest")
    print(f"  Anomaly (Unknown)   : {unknown}   ← caught by Autoencoder (Zero-Day)")
