"""Validation Strategy Recommender choosing appropriate splitting schemes."""

from typing import Optional
import pandas as pd

from amea.core.state import TaskType


class ValidationRecommender:
    """Recommends cross-validation schemes to prevent data leakage and handle imbalance."""

    @staticmethod
    def recommend(
        task_type: TaskType,
        df: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
    ) -> str:
        if task_type == TaskType.TIME_SERIES:
            return "TimeSeriesSplit"

        if df is not None:
            # Check for temporal columns
            date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in c.lower() or "timestamp" in c.lower()]
            if date_cols and len(date_cols) > 0:
                return "TimeSeriesSplit"

        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            return "StratifiedKFold"

        return "KFold"
