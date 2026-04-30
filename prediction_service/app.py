"""
app.py
------
FastAPI entry point for the MLOps Prediction Service.
Responsibilities: define routes, validate input, return predictions.
Model loading and class blueprints are handled by model_loader.py.

Prediction labels:
  0 → normal
  1 → anomaly (known)
  2 → anomaly (unknown / Zero-Day)
"""
import io

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Any

from model_loader import load_artifacts
from preprocess import preprocess_input, COLUMNS
from utils import log_prediction

app = FastAPI(title="MLOps Prediction Service")

# Load pre-trained artifacts once at startup
model, preprocessor = load_artifacts()

LABEL_MAP = {
    0: "normal",
    1: "confirmed threat",   # RF detected a known attack signature
    2: "novel threat"        # Autoencoder flagged a potential Zero-Day
}


class PredictionRequest(BaseModel):
    features: List[Any]


@app.get("/health")
def health():
    return {"status": "ok", "service": "prediction", "port": 8000}


@app.post("/predict")
def predict(request: PredictionRequest):
    """Single-record prediction endpoint."""
    if not model or not preprocessor:
        raise HTTPException(status_code=500, detail="Models not loaded")

    features = request.features
    if len(features) != len(COLUMNS):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(COLUMNS)} features, got {len(features)}"
        )

    try:
        X_p = preprocess_input(features, preprocessor)
        pred = int(model.predict(X_p)[0])
        label = LABEL_MAP.get(pred, "unknown")
        # anomaly flag: 0 = normal, 1 = any anomaly (for dashboard compatibility)
        anomaly = 0 if pred == 0 else 1
        log_prediction(features, anomaly)
        return {"anomaly": anomaly, "label": label, "code": pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """Batch prediction endpoint — accepts a CSV file without headers."""
    if not model or not preprocessor:
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), header=None)

        if df.shape[1] > len(COLUMNS):
            df = df.iloc[:, :len(COLUMNS)]

        if df.shape[1] < len(COLUMNS):
            raise HTTPException(
                status_code=400,
                detail=f"Expected at least {len(COLUMNS)} features, got {df.shape[1]}"
            )

        df.columns = COLUMNS
        X_p = preprocessor.transform(df)
        preds = model.predict(X_p).tolist()
        labels = [LABEL_MAP.get(p, "unknown") for p in preds]
        anomalies = [0 if p == 0 else 1 for p in preds]

        for idx, row in df.iterrows():
            log_prediction(row.tolist(), anomalies[idx])

        return {"predictions": anomalies, "labels": labels, "codes": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/reload")
def reload_model():
    """
    Hot-reload model artifacts from disk without restarting the server.
    Called automatically by the drift service after retraining completes.
    """
    global model, preprocessor
    model, preprocessor = load_artifacts()
    if model and preprocessor:
        return {"status": "reloaded", "message": "Model artifacts reloaded successfully."}
    raise HTTPException(status_code=500, detail="Failed to reload model artifacts.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
