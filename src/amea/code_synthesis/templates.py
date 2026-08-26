"""Evidence-driven pipeline template generators."""

import json
from pathlib import Path
from typing import Any, Dict
from amea.code_synthesis.models import CodeSynthesisContext
from amea.core.state import TaskType


class PipelineTemplateEngine:
    """Renders modular, self-contained, production-quality Python ML pipeline files."""

    @staticmethod
    def render_data_loader(context: CodeSynthesisContext) -> str:
        target_col = context.task_spec.target_column
        dataset_path = context.data_profile.dataset_path if context.data_profile else "data.csv"
        clean_path = Path(dataset_path).as_posix()

        return f'''"""Data loader module for raw dataset ingestion."""

import os
from pathlib import Path
from typing import Tuple
import pandas as pd


def load_dataset(file_path: str = r"{clean_path}") -> pd.DataFrame:
    """Load and validate raw dataset."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {{file_path}}")
    
    df = pd.read_csv(path)
    target = "{target_col}"
    if target not in df.columns:
        raise ValueError(f"Target column '{{target}}' not found in dataset columns: {{list(df.columns)}}")
    
    return df


def split_features_target(df: pd.DataFrame, target_column: str = "{target_col}") -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y."""
    X = df.drop(columns=[target_column], errors="ignore")
    y = df[target_column]
    return X, y
'''

    @staticmethod
    def render_preprocess(context: CodeSynthesisContext) -> str:
        # Determine if scaling is needed based on best model family
        model_family = context.best_candidate.model_family
        requires_scaling = model_family in ["LinearModel", "TabularNeuralNet"]

        return f'''"""Data preprocessing module for cleaning and imputation."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class PipelinePreprocessor(BaseEstimator, TransformerMixin):
    """Reproducible preprocessing pipeline fitted strictly on training data."""

    def __init__(self, requires_scaling: bool = {requires_scaling}):
        self.requires_scaling = requires_scaling
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler() if requires_scaling else None
        self.numeric_columns = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit imputation and optional scaling on numeric features."""
        numeric_df = X.select_dtypes(include=[np.number])
        self.numeric_columns = list(numeric_df.columns)
        
        if self.numeric_columns:
            imputed = self.imputer.fit_transform(numeric_df)
            if self.scaler is not None:
                self.scaler.fit(imputed)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted transformations."""
        if not self.numeric_columns:
            return np.empty((len(X), 0))
            
        numeric_df = X[self.numeric_columns]
        imputed = self.imputer.transform(numeric_df)
        if self.scaler is not None:
            return self.scaler.transform(imputed)
        return imputed
'''

    @staticmethod
    def render_features(context: CodeSynthesisContext) -> str:
        return '''"""Feature engineering module for domain-specific feature derivation."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Encapsulates feature transformations and non-linear interactions."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Pass through preprocessed features with support for expanded feature spaces."""
        return np.asarray(X)
'''

    @staticmethod
    def render_train(context: CodeSynthesisContext) -> str:
        candidate = context.best_candidate
        task_type = context.task_spec.task_type
        is_classification = (task_type != TaskType.REGRESSION and task_type != TaskType.TIME_SERIES)
        seed = 42
        params_json = json.dumps(candidate.hyperparameters)

        # Determine import and class
        cls_name = candidate.hyperparameters.get("model_class_name", "")
        if not cls_name:
            if candidate.model_family == "LinearModel":
                cls_name = "LogisticRegression" if is_classification else "Ridge"
            elif candidate.model_family == "RandomForest":
                cls_name = "RandomForestClassifier" if is_classification else "RandomForestRegressor"
            elif candidate.model_family == "GradientBoosting":
                cls_name = "HistGradientBoostingClassifier" if is_classification else "HistGradientBoostingRegressor"
            elif candidate.model_family == "TabularNeuralNet":
                cls_name = "MLPClassifier" if is_classification else "MLPRegressor"
            else:
                cls_name = "LogisticRegression" if is_classification else "Ridge"

        # Import mapping
        if cls_name in ["LogisticRegression", "Ridge"]:
            import_stmt = f"from sklearn.linear_model import {cls_name}"
        elif cls_name in ["RandomForestClassifier", "RandomForestRegressor", "HistGradientBoostingClassifier", "HistGradientBoostingRegressor"]:
            import_stmt = f"from sklearn.ensemble import {cls_name}"
        elif cls_name in ["MLPClassifier", "MLPRegressor"]:
            import_stmt = f"from sklearn.neural_network import {cls_name}"
        else:
            import_stmt = "from sklearn.linear_model import LogisticRegression"
            cls_name = "LogisticRegression"

        return f'''"""Model training module."""

import json
from pathlib import Path
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
{import_stmt}

from data_loader import load_dataset, split_features_target
from preprocess import PipelinePreprocessor
from features import FeatureTransformer
from evaluate import compute_metrics


def build_pipeline() -> Pipeline:
    """Construct end-to-end scikit-learn Pipeline."""
    model = {cls_name}(**{params_json})
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
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state={seed}) if '{is_classification}' == 'True' else KFold(n_splits=3, shuffle=True, random_state={seed})
    fold_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_val)
        proba = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") and '{is_classification}' == 'True' else None
        metrics = compute_metrics(y_val, preds, proba)
        fold_scores.append(metrics)

    # Fit final pipeline on all data
    pipeline.fit(X, y)
    
    # Save model artifact
    artifact_path = Path("model.joblib")
    joblib.dump(pipeline, artifact_path)
    
    return {{"cross_validation_metrics": fold_scores, "model_artifact": str(artifact_path)}}


if __name__ == "__main__":
    results = train_and_evaluate()
    if results and "cross_validation_metrics" in results:
        avg_metrics = {{}}
        folds = results["cross_validation_metrics"]
        for fold in folds:
            for k, v in fold.items():
                avg_metrics[k] = avg_metrics.get(k, 0.0) + (v / len(folds))
        print(f"__AMEA_METRICS__={{json.dumps(avg_metrics)}}")
'''

    @staticmethod
    def render_evaluate(context: CodeSynthesisContext) -> str:
        is_classification = (context.task_spec.task_type != TaskType.REGRESSION and context.task_spec.task_type != TaskType.TIME_SERIES)
        primary_metric = context.task_spec.primary_metric

        return f'''"""Evaluation metric calculation module."""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score


def compute_metrics(y_true: Any, y_pred: Any, y_proba: Optional[Any] = None) -> Dict[str, float]:
    """Calculate primary and secondary evaluation metrics."""
    metrics: Dict[str, float] = {{}}
    
    if '{is_classification}' == 'True':
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
'''

    @staticmethod
    def render_inference(context: CodeSynthesisContext) -> str:
        return '''"""Production inference module."""

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
'''

    @staticmethod
    def render_requirements(context: CodeSynthesisContext) -> str:
        return '''numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
'''

    @staticmethod
    def render_config(context: CodeSynthesisContext, pipeline_id: str) -> str:
        config = {
            "pipeline_id": pipeline_id,
            "task_type": context.task_spec.task_type.value,
            "target_column": context.task_spec.target_column,
            "primary_metric": context.task_spec.primary_metric,
            "best_model_family": context.best_candidate.model_family,
            "hyperparameters": context.best_candidate.hyperparameters,
            "training_metrics": context.best_candidate.cv_metrics_mean,
            "inference_latency_ms": context.best_candidate.inference_latency_ms,
        }
        return json.dumps(config, indent=2)
