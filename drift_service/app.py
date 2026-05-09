"""
app.py
------
FastAPI microservice for drift detection.
Runs on port 8001.

Endpoints:
  GET  /health          - Health check
  GET  /drift/status    - Quick status (stable / drift_detected / error)
  GET  /drift/report    - Full drift report with per-feature KS details
  POST /drift/retrain   - Trigger ml_pipeline retraining + hot-reload model
"""

import os
import subprocess

import requests as http_requests
from fastapi import FastAPI, BackgroundTasks

from detector import run_drift_check

app = FastAPI(title="Drift Detection Service", version="1.0.0")

BASE_DIR = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ML_PIPELINE_DIR = os.path.join(BASE_DIR, "ml_pipeline")
PREDICTION_SERVICE_URL = os.environ.get("PREDICTION_SERVICE_URL", "http://localhost:8000")


@app.get("/health")
def health():
    return {"status": "ok", "service": "drift-detection", "port": 8001}


@app.get("/drift/status")
def drift_status(window: int = 1000):
    """
    Quick drift status check.
    Returns overall_drift flag without full feature-level detail.
    """
    report = run_drift_check(window)
    return {
        "status": report["status"],
        "timestamp": report["timestamp"],
        "log_count": report.get("log_count", 0),
        "overall_drift": report.get("overall_drift", False),
        "anomaly_rate": report.get("anomaly_rate_check", {}).get("recent_rate"),
        "baseline_anomaly_rate": report.get("anomaly_rate_check", {}).get("baseline_rate"),
        "drifted_features": report.get("feature_drift_check", {}).get("drifted_features", []),
    }


@app.get("/drift/report")
def drift_report(window: int = 1000):
    """
    Full detailed drift report including per-feature KS statistics.
    """
    return run_drift_check(window)


@app.post("/drift/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Trigger model retraining in the background.
    After training completes, hot-reloads the prediction service via POST /reload.
    """
    background_tasks.add_task(_run_retraining)
    return {
        "status": "retraining_started",
        "message": "ML pipeline triggered in background. Check server logs for progress.",
    }


def _run_retraining():
    """
    Runs ml_pipeline/retrain.py (initial 70% + reserve 30%) as a subprocess,
    then signals the prediction service to hot-reload the new .pkl artifacts.
    """
    print("[Drift Service] Starting drift-triggered retraining (full dataset)...")
    result = subprocess.run(
        ["python", "retrain.py"],
        cwd=ML_PIPELINE_DIR,
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    if result.returncode == 0:
        print("[Drift Service] Retraining complete. Hot-reloading prediction service...")
        try:
            resp = http_requests.post(f"{PREDICTION_SERVICE_URL}/reload", timeout=30)
            if resp.status_code == 200:
                print("[Drift Service] Prediction service reloaded successfully.")
            else:
                print(f"[Drift Service] Reload returned {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[Drift Service] Could not reach prediction service for reload: {e}")
    else:
        print(f"[Drift Service] Retraining FAILED:\n{result.stderr}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
