"""Model training module."""

import json
from pathlib import Path
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_dataset, split_features_target
from preprocess import PipelinePreprocessor
from features import FeatureTransformer
from evaluate import compute_metrics


def build_pipeline() -> Pipeline:
    """Construct end-to-end scikit-learn Pipeline."""
    model = RandomForestClassifier(**{"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "random_state": 42})
    pipeline = Pipeline([
        ("preprocessor", PipelinePreprocessor()),
        ("features", FeatureTransformer()),
        ("model", model),
    ])
    return pipeline


def train_and_evaluate(dataset_path: str = None) -> Dict[str, Any]:
    """Train pipeline, evaluate via cross-validation, and serialize artifacts."""
    df = load_dataset(dataset_path) if dataset_path else load_dataset()
    X, y = split_features_target(df)

    pipeline = build_pipeline()
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) if 'True' == 'True' else KFold(n_splits=3, shuffle=True, random_state=42)
    fold_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_val)
        proba = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") and 'True' == 'True' else None
        metrics = compute_metrics(y_val, preds, proba)
        fold_scores.append(metrics)

    # Fit final pipeline on all data
    pipeline.fit(X, y)
    
    # Save model artifact
    artifact_path = Path("model.joblib")
    joblib.dump(pipeline, artifact_path)
    
    return {"cross_validation_metrics": fold_scores, "model_artifact": str(artifact_path)}


if __name__ == "__main__":
    results = train_and_evaluate()
    if results and "cross_validation_metrics" in results:
        avg_metrics = {}
        folds = results["cross_validation_metrics"]
        for fold in folds:
            for k, v in fold.items():
                avg_metrics[k] = avg_metrics.get(k, 0.0) + (v / len(folds))
        print(f"__AMEA_METRICS__={json.dumps(avg_metrics)}")
