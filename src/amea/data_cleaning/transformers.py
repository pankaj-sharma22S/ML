"""Reproducible, scikit-learn compatible data cleaning transformers."""

from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDropperTransformer(BaseEstimator, TransformerMixin):
    """Drops specified columns during transform."""

    def __init__(self, columns_to_drop: Optional[List[str]] = None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols_present = [c for c in self.columns_to_drop if c in X_df.columns]
        if cols_present:
            return X_df.drop(columns=cols_present)
        return X_df


class OutlierClipperTransformer(BaseEstimator, TransformerMixin):
    """Clips numeric outliers to quantiles learned strictly during fit."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99, columns: Optional[List[str]] = None):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.columns = columns
        self.bounds_: Dict[str, tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        target_cols = self.columns or [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]

        self.bounds_ = {}
        for col in target_cols:
            if col in X_df.columns:
                series = X_df[col].dropna()
                if len(series) > 0:
                    low = float(series.quantile(self.lower_quantile))
                    high = float(series.quantile(self.upper_quantile))
                    self.bounds_[col] = (low, high)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col, (low, high) in self.bounds_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower=low, upper=high)
        return X_df


class RareCategoryGrouperTransformer(BaseEstimator, TransformerMixin):
    """Groups rare categories and unseen test categories into '__OTHER__'."""

    def __init__(self, min_frequency: float = 0.01, columns: Optional[List[str]] = None, other_value: str = "__OTHER__"):
        self.min_frequency = min_frequency
        self.columns = columns
        self.other_value = other_value
        self.frequent_categories_: Dict[str, Set[str]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        target_cols = self.columns or [c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])]

        self.frequent_categories_ = {}
        total_rows = len(X_df)

        for col in target_cols:
            if col in X_df.columns and total_rows > 0:
                val_counts = X_df[col].dropna().value_counts(normalize=True)
                frequent = set(val_counts[val_counts >= self.min_frequency].index.astype(str))
                self.frequent_categories_[col] = frequent
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col, frequent_set in self.frequent_categories_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].astype(str).apply(lambda val: val if val in frequent_set else self.other_value)
        return X_df


class AdaptiveImputerTransformer(BaseEstimator, TransformerMixin):
    """Imputes numeric features with median and categorical features with mode learned on train set."""

    def __init__(self):
        self.numeric_medians_: Dict[str, float] = {}
        self.categorical_modes_: Dict[str, str] = {}

    def fit(self, X: pd.DataFrame, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.numeric_medians_ = {}
        self.categorical_modes_ = {}

        for col in X_df.columns:
            series = X_df[col].dropna()
            if len(series) > 0:
                if pd.api.types.is_numeric_dtype(series):
                    self.numeric_medians_[col] = float(series.median())
                else:
                    mode_val = series.mode()
                    if not mode_val.empty:
                        self.categorical_modes_[col] = str(mode_val.iloc[0])
                    else:
                        self.categorical_modes_[col] = "UNKNOWN"
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col, med in self.numeric_medians_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(med)

        for col, mode_val in self.categorical_modes_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(mode_val)

        return X_df
