"""Data preprocessing module for cleaning and imputation."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class PipelinePreprocessor(BaseEstimator, TransformerMixin):
    """Reproducible preprocessing pipeline fitted strictly on training data."""

    def __init__(self, requires_scaling: bool = False):
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
