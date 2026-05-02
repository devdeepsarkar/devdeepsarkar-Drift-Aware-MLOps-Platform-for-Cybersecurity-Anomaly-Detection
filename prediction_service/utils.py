import os
import csv
from datetime import datetime

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_FILE = os.path.join(PROJECT_ROOT, "data_storage", "logs.csv")

def log_prediction(features, prediction):
    """
    Logs the prediction to data_storage/logs.csv.
    Format: features..., prediction, timestamp
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        row = list(features) + [prediction, datetime.utcnow().isoformat()]
        writer.writerow(row)
