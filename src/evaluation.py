# src/evaluation.py
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from typing import Dict

def run(model, df, metrics):
    """
    Evaluate 'model' on df (expects df contains 'target' and numeric features).
    Returns a dict of metrics.
    """
    print("📊 Evaluating model...")
    X = df.drop(columns=["target"], errors="ignore").select_dtypes(include="number")
    y = df["target"].values

    if model is None:
        print("⚠️ Model is None -> returning None metrics.")
        return {m: None for m in metrics}

    y_pred = model.predict(X)
    results: Dict[str, float] = {}
    for m in metrics:
        if m.lower() == "rmse":
            results[m] = float(mean_squared_error(y, y_pred, squared=False))
        elif m.lower() == "mae":
            results[m] = float(mean_absolute_error(y, y_pred))
        elif m.lower() in ("mape", "MAPE"):
            # avoid division by zero
            denom = np.where(y == 0, 1e-8, y)
            results[m] = float(np.mean(np.abs((y - y_pred) / denom)) * 100.0)
        else:
            results[m] = None
    return results

def save_metrics(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📄 Metrics saved to: {path}")
