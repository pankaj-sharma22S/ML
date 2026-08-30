"""Evaluation metric calculation module."""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score


def compute_metrics(y_true: Any, y_pred: Any, y_proba: Optional[Any] = None) -> Dict[str, float]:
    """Calculate primary and secondary evaluation metrics."""
    metrics: Dict[str, float] = {}
    
    if 'True' == 'True':
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        try:
            metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted"))
        except Exception:
            metrics["f1"] = 0.0
            
        if y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except Exception:
                metrics["roc_auc"] = metrics["accuracy"]
    else:
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["r2"] = float(r2_score(y_true, y_pred))

    return metrics
