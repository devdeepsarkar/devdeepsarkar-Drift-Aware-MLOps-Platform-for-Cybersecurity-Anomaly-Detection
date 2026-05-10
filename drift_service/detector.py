"""
detector.py
-----------
Drift detection logic for the CyberSecurity Log Anomaly Detection System.

Two drift checks:
  1. Anomaly Rate Drift  — monitors if the fraction of anomalies in recent
     predictions deviates significantly from the training baseline.
  2. Feature Drift (KS-Test) — uses the Kolmogorov-Smirnov test to detect
     if the distribution of numeric features in recent logs differs from
     the training baseline distribution.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

BASE_DIR = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_PATH = os.path.join(BASE_DIR, "data_storage", "logs.csv")
BASELINE_PATH = os.path.join(BASE_DIR, "data_storage", "baseline_stats.json")

FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

# Only numeric columns are used for KS-test drift
NUMERIC_COLUMNS = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]


def load_logs(window: int = 1000) -> pd.DataFrame | None:
    """Load the last `window` rows from logs.csv."""
    if not os.path.exists(LOGS_PATH):
        return None
    cols = FEATURE_COLUMNS + ["prediction", "timestamp"]
    try:
        df = pd.read_csv(LOGS_PATH, header=None, names=cols)
        return df.tail(window)
    except Exception:
        return None


def load_baseline() -> dict | None:
    """Load baseline statistics saved during training."""
    if not os.path.exists(BASELINE_PATH):
        return None
    with open(BASELINE_PATH) as f:
        return json.load(f)


def check_anomaly_rate_drift(
    df: pd.DataFrame, baseline: dict, threshold: float = 0.15
) -> dict:
    """
    Compare the recent anomaly rate against the training baseline.
    Flags drift if |recent_rate - baseline_rate| > threshold.
    """
    recent_rate = (pd.to_numeric(df["prediction"], errors="coerce") > 0).mean()
    baseline_rate = baseline["anomaly_rate"]
    delta = abs(float(recent_rate) - baseline_rate)
    drift = delta > threshold
    return {
        "type": "anomaly_rate",
        "baseline_rate": round(baseline_rate, 4),
        "recent_rate": round(float(recent_rate), 4),
        "delta": round(delta, 4),
        "threshold": threshold,
        "drift_detected": bool(drift),
    }


def check_feature_drift(
    df: pd.DataFrame, baseline: dict, p_threshold: float = 0.05
) -> dict:
    """
    Run KS-test on numeric features comparing recent logs vs training
    baseline distributions (reconstructed from saved mean/std).
    """
    feature_results = {}
    any_drift = False
    baseline_means = baseline.get("feature_means", {})
    baseline_stds = baseline.get("feature_stds", {})
    rng = np.random.default_rng(42)

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        try:
            recent_vals = pd.to_numeric(df[col], errors="coerce").dropna().values
            if len(recent_vals) < 30:
                continue
            b_mean = baseline_means.get(col)
            b_std = baseline_stds.get(col)
            if b_mean is None or b_std is None or float(b_std) == 0:
                continue
            # Reconstruct a baseline sample using saved mean/std
            baseline_sample = rng.normal(float(b_mean), float(b_std), size=len(recent_vals))
            ks_stat, p_value = stats.ks_2samp(recent_vals, baseline_sample)
            drifted = bool(p_value < p_threshold)
            if drifted:
                any_drift = True
            feature_results[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": drifted,
            }
        except Exception:
            continue

    return {
        "type": "feature_ks_test",
        "p_value_threshold": p_threshold,
        "drift_detected": any_drift,
        "drifted_features": [k for k, v in feature_results.items() if v["drift_detected"]],
        "feature_details": feature_results,
    }


def run_drift_check(window: int = 1000) -> dict:
    """Run all drift checks and return a unified report."""
    timestamp = datetime.utcnow().isoformat()

    baseline = load_baseline()
    if baseline is None:
        return {
            "timestamp": timestamp,
            "status": "error",
            "message": "Baseline stats not found. Run ml_pipeline/main.py to generate them.",
        }

    df = load_logs(window)
    if df is None or len(df) < 50:
        return {
            "timestamp": timestamp,
            "status": "insufficient_data",
            "message": f"Need at least 50 log entries. Found: {0 if df is None else len(df)}",
            "log_count": 0 if df is None else len(df),
        }

    anomaly_check = check_anomaly_rate_drift(df, baseline)
    feature_check = check_feature_drift(df, baseline)
    overall_drift = anomaly_check["drift_detected"] or feature_check["drift_detected"]

    return {
        "timestamp": timestamp,
        "status": "drift_detected" if overall_drift else "stable",
        "log_count": len(df),
        "window": window,
        "overall_drift": overall_drift,
        "anomaly_rate_check": anomaly_check,
        "feature_drift_check": {
            "type": feature_check["type"],
            "p_value_threshold": feature_check["p_value_threshold"],
            "drift_detected": feature_check["drift_detected"],
            "drifted_features": feature_check["drifted_features"],
            "drifted_feature_count": len(feature_check["drifted_features"]),
            "feature_details": feature_check["feature_details"],
        },
    }
