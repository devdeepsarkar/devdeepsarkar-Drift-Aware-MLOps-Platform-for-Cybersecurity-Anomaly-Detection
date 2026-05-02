import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer that applies domain-specific feature
    engineering to the NSL-KDD cybersecurity dataset:

    1. Log Transform — Applies np.log1p() to src_bytes and dst_bytes to tame
       extreme outliers (e.g., large file downloads skewing the scale).
    2. Byte Ratio — Creates a new 'byte_ratio' feature (dst_bytes / src_bytes)
       that acts as a "data exfiltration" signal for anomaly detection.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a copy to avoid SettingWithCopyWarning
        X_new = X.copy()
        if not isinstance(X_new, pd.DataFrame):
            X_new = pd.DataFrame(X_new)

        # 1. Log Transform byte columns
        if 'src_bytes' in X_new.columns:
            X_new['src_bytes'] = np.log1p(X_new['src_bytes'].astype(float))
        if 'dst_bytes' in X_new.columns:
            X_new['dst_bytes'] = np.log1p(X_new['dst_bytes'].astype(float))

        # 2. Feature Interaction: Ratio of Outgoing to Incoming bytes
        if 'src_bytes' in X_new.columns and 'dst_bytes' in X_new.columns:
            # Add 1 to avoid division by zero
            X_new['byte_ratio'] = X_new['dst_bytes'].astype(float) / (X_new['src_bytes'].astype(float) + 1.0)

        return X_new


