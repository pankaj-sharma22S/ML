"""Metric Recommender aligning business objectives with statistical properties."""

from typing import List, Optional, Tuple
import pandas as pd

from amea.core.state import TaskType
from amea.problem_understanding.models import IntentAnalysis


class MetricRecommender:
    """Selects primary optimization metric and secondary evaluation metrics deterministically."""

    @staticmethod
    def recommend(
        task_type: TaskType,
        intent: IntentAnalysis,
        df: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
    ) -> Tuple[str, List[str], str]:
        # 1. User Explicit Metric Override
        if intent.requested_metrics:
            user_metric = intent.requested_metrics[0]
            direction = "minimize" if user_metric in ["rmse", "mae", "logloss", "loss"] else "maximize"
            secondaries = [m for m in intent.requested_metrics[1:]]
            return user_metric, secondaries, direction

        # 2. Imbalance-Aware Classification Metrics
        if task_type == TaskType.BINARY_CLASSIFICATION:
            is_imbalanced = False
            if df is not None and target_column and target_column in df.columns:
                counts = df[target_column].dropna().value_counts(normalize=True)
                if len(counts) > 1 and counts.iloc[-1] <= 0.15:
                    is_imbalanced = True

            if is_imbalanced:
                return "pr_auc", ["roc_auc", "f1_macro", "accuracy"], "maximize"
            return "roc_auc", ["accuracy", "f1", "precision", "recall"], "maximize"

        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return "f1_macro", ["accuracy", "logloss", "balanced_accuracy"], "maximize"

        elif task_type in (TaskType.REGRESSION, TaskType.TIME_SERIES):
            return "rmse", ["mae", "r2", "mape"], "minimize"

        # Default fallback
        return "accuracy", ["f1"], "maximize"
