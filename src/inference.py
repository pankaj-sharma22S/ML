"""Production inference module."""

from pathlib import Path
from typing import Any, Dict, Union
import joblib
import numpy as np
import pandas as pd


class ModelPredictor:
    """Loads serialized pipeline and performs real-time or batch inference."""

    def __init__(self, model_artifact_path: str = "model.joblib"):
        path = Path(model_artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_artifact_path}")
        self.pipeline = joblib.load(path)

    def predict(self, data: Union[pd.DataFrame, Dict[str, Any], list]) -> Dict[str, Any]:
        """Generate predictions and optional probabilities for new input data."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        predictions = self.pipeline.predict(df)
        result = {"predictions": predictions.tolist()}

        if hasattr(self.pipeline, "predict_proba"):
            try:
                probabilities = self.pipeline.predict_proba(df)
                result["probabilities"] = probabilities.tolist()
            except Exception:
                pass

        return result
