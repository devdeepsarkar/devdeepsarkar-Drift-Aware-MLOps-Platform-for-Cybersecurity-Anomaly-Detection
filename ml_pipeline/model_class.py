"""
model_class.py
--------------
Defines the HybridAnomalyDetector — a two-stage ensemble model that combines:

  1. Random Forest (Supervised) — detects KNOWN attack patterns.
  2. Autoencoder / MLPRegressor (Unsupervised) — detects UNKNOWN / Zero-Day attacks
     by learning what "normal" traffic looks like and flagging high reconstruction error.

Prediction outputs:
  0 → Normal
  1 → Anomaly (Known)   — RF flagged a recognised attack signature
  2 → Anomaly (Unknown) — Autoencoder flagged a Zero-Day / novel threat
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, ClassifierMixin

LABEL_NORMAL = 0
LABEL_KNOWN = 1
LABEL_UNKNOWN = 2


class HybridAnomalyDetector(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        ae_hidden_layer_sizes=(64, 32, 64),
        ae_contamination=0.05,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.ae_hidden_layer_sizes = ae_hidden_layer_sizes
        self.ae_contamination = ae_contamination
        self.random_state = random_state

        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=self.ae_hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=self.random_state,
            early_stopping=True
        )
        self.ae_threshold_ = None

    def fit(self, X, y):
        # --- Stage 1: Train Random Forest on all labelled data ---
        print("Training Random Forest (known attack detector)...")
        self.rf.fit(X, y)
        print("Random Forest training complete!")

        # --- Stage 2: Train Autoencoder ONLY on normal traffic ---
        print("Training Autoencoder on normal traffic only (Zero-Day detector)...")
        X_normal = X[y == 0]
        self.autoencoder.fit(X_normal, X_normal)

        # Calculate reconstruction error threshold on normal data
        X_pred = self.autoencoder.predict(X_normal)
        mse = np.mean(np.power(X_normal - X_pred, 2), axis=1)
        # Flag top ae_contamination % of reconstruction errors as suspicious
        self.ae_threshold_ = np.percentile(mse, 100 * (1 - self.ae_contamination))
        print(f"Autoencoder training complete! MSE threshold: {self.ae_threshold_:.6f}")
        return self

    def predict(self, X):
        """
        Hybrid logic:
          1. RF predicts first. If it says anomaly → KNOWN (1).
          2. Otherwise Autoencoder checks. If MSE > threshold → UNKNOWN (2).
          3. Both say safe → NORMAL (0).
        """
        rf_preds = self.rf.predict(X)

        X_reconstructed = self.autoencoder.predict(X)
        mse = np.mean(np.power(X - X_reconstructed, 2), axis=1)

        results = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            if rf_preds[i] == 1:
                results[i] = LABEL_KNOWN       # RF caught a known attack
            elif mse[i] > self.ae_threshold_:
                results[i] = LABEL_UNKNOWN     # Autoencoder caught a Zero-Day
            else:
                results[i] = LABEL_NORMAL      # Both say safe
        return results
