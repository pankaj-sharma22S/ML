"""Configuration options for interactive Python execution kernels."""

from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class KernelConfig(BaseModel):
    """Resource, security, and execution limits for an interactive Python kernel session."""
    kernel_name: str = "python3"
    default_timeout_seconds: int = 120
    max_memory_mb: int = 4096
    max_cpu_percent: float = 80.0
    max_dataframe_rows: int = 100
    max_output_size_bytes: int = 2 * 1024 * 1024  # 2MB max stdout/output buffer
    auto_restart_on_crash: bool = True
    base_workspace_dir: Optional[Path] = Field(default_factory=lambda: Path(".amea_project/sessions").resolve())
    allowed_packages: List[str] = Field(
        default_factory=lambda: [
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "sklearn",
            "joblib",
            "polars",
            "lightgbm",
            "xgboost",
            "torch",
            "torchvision",
            "matplotlib",
            "seaborn",
            "statsmodels",
            "optuna",
        ]
    )
