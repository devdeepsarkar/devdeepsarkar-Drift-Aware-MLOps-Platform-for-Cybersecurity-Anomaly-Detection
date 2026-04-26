"""
train.py
--------
Thin training wrapper. Imports HybridAnomalyDetector from model_class
so the saved .pkl file references 'model_class.HybridAnomalyDetector'.
"""
from model_class import HybridAnomalyDetector  # noqa: F401


def train_model(X_train, y_train):
    model = HybridAnomalyDetector(
        n_estimators=100,
        ae_hidden_layer_sizes=(64, 32, 64),
        ae_contamination=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
