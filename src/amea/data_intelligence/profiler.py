"""Deterministic deep statistical profiler."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from amea.core.state import ColumnProfile, DataProfile


class DeepDataProfiler:
    """Profiles tabular datasets with deterministic mathematical guarantees."""

    @staticmethod
    def profile(df: pd.DataFrame, dataset_path: str, sha256: str) -> DataProfile:
        """Compute full statistical profile of DataFrame."""
        total_rows = len(df)
        total_cols = len(df.columns)
        col_profiles: Dict[str, ColumnProfile] = {}
        leakage_suspects: List[str] = []

        for col in df.columns:
            series = df[col]
            null_count = int(series.isnull().sum())
            null_ratio = float(null_count / total_rows) if total_rows > 0 else 0.0
            distinct_count = int(series.nunique(dropna=True))
            is_const = (distinct_count <= 1)
            is_uniq = (distinct_count == total_rows and total_rows > 0)

            # ID candidate heuristic
            if is_uniq and (pd.api.types.is_integer_dtype(series) or "id" in col.lower() or "key" in col.lower()):
                leakage_suspects.append(col)

            mean_val = None
            std_val = None
            min_val = None
            max_val = None
            skew_val = None
            class_bal = None

            if pd.api.types.is_numeric_dtype(series):
                valid_vals = series.dropna().values
                if len(valid_vals) > 0:
                    mean_val = float(np.mean(valid_vals))
                    std_val = float(np.std(valid_vals))
                    min_val = float(np.min(valid_vals))
                    max_val = float(np.max(valid_vals))
                    if len(valid_vals) > 2 and std_val > 1e-9:
                        skew_val = float(stats.skew(valid_vals))
            else:
                # Categorical class distribution for low cardinality
                if distinct_count <= 20 and total_rows > 0:
                    val_counts = series.value_counts(normalize=True, dropna=True).to_dict()
                    class_bal = {str(k): round(float(v), 4) for k, v in val_counts.items()}

            col_profiles[col] = ColumnProfile(
                dtype=str(series.dtype),
                null_count=null_count,
                null_ratio=round(null_ratio, 4),
                distinct_count=distinct_count,
                is_constant=is_const,
                is_unique=is_uniq,
                mean=round(mean_val, 4) if mean_val is not None else None,
                std=round(std_val, 4) if std_val is not None else None,
                min=round(min_val, 4) if min_val is not None else None,
                max=round(max_val, 4) if max_val is not None else None,
                skewness=round(skew_val, 4) if skew_val is not None else None,
                class_balance=class_bal,
            )

        memory_mb = round(float(df.memory_usage(deep=True).sum() / (1024 ** 2)), 2)
        duplicate_rows = int(df.duplicated().sum())

        return DataProfile(
            dataset_path=dataset_path,
            dataset_sha256=sha256,
            total_rows=total_rows,
            total_columns=total_cols,
            columns=col_profiles,
            memory_footprint_mb=memory_mb,
            duplicate_rows=duplicate_rows,
            potential_leakage_columns=leakage_suspects,
        )
