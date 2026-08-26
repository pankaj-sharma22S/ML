"""Statistical profiling and candidate key discovery for ingested datasets."""

from typing import Dict, List
import numpy as np
import pandas as pd
from amea.query_analysis.schemas import DatasetProfile
from amea.query_analysis.ingestion import IngestionFileRecord


class DataProfilerEngine:
    """Computes comprehensive dataset profiles including key and datetime candidates."""

    @classmethod
    def profile_dataset(cls, record: IngestionFileRecord) -> DatasetProfile:
        df = record.df
        rows, cols = df.shape

        col_names = list(df.columns)
        col_types: Dict[str, str] = {c: str(df[c].dtype) for c in col_names}

        numeric_cols: List[str] = list(df.select_dtypes(include=[np.number]).columns)
        categorical_cols: List[str] = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
        datetime_cols: List[str] = []

        # Detect datetime candidates in categorical columns
        for c in categorical_cols:
            sample = df[c].dropna().astype(str).head(20)
            if not sample.empty:
                try:
                    # Attempt parse sample
                    pd.to_datetime(sample, errors="raise")
                    datetime_cols.append(c)
                except Exception:
                    pass

        # Identify candidate keys (100% unique & non-null)
        candidate_keys: List[str] = []
        for c in col_names:
            if df[c].nunique() == rows and df[c].isnull().sum() == 0:
                candidate_keys.append(c)

        # Missing values summary
        missing_summary: Dict[str, int] = df.isnull().sum().to_dict()
        dup_rows = int(df.duplicated().sum())

        # Summary stats for numeric columns
        summary_stats: Dict[str, Dict[str, float]] = {}
        for num_c in numeric_cols:
            series = df[num_c].dropna()
            if not series.empty:
                summary_stats[num_c] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()) if len(series) > 1 else 0.0,
                    "min": float(series.min()),
                    "median": float(series.median()),
                    "max": float(series.max()),
                }

        return DatasetProfile(
            dataset_id=record.dataset_id,
            original_filename=record.filename,
            file_type=record.file_type,
            file_size_bytes=record.file_size_bytes,
            rows=rows,
            columns=cols,
            column_names=col_names,
            column_types=col_types,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            candidate_keys=candidate_keys,
            duplicate_rows_count=dup_rows,
            missing_summary=missing_summary,
            summary_stats=summary_stats,
        )
