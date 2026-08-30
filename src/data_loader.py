"""Data loader module for raw dataset ingestion."""

import os
from pathlib import Path
from typing import Tuple
import pandas as pd


def load_dataset(file_path: str = r"D:/ML/data/sample_churn.csv") -> pd.DataFrame:
    """Load and validate raw dataset."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(path)
    target = "churn"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns: {list(df.columns)}")
    
    return df


def split_features_target(df: pd.DataFrame, target_column: str = "churn") -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y."""
    X = df.drop(columns=[target_column], errors="ignore")
    y = df[target_column]
    return X, y
