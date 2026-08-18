"""Deterministic system discovery tool inspecting hardware and software capabilities."""

import os
import platform
import sys
from typing import Dict
from pydantic import BaseModel, Field


class HardwareCapabilities(BaseModel):
    """Factual, detected hardware resources (strictly separated from user-authorized budget)."""
    os_platform: str
    python_version: str
    cpu_cores: int
    system_ram_gb: float
    gpu_available: bool = False
    gpu_device_count: int = 0
    gpu_device_name: str | None = None
    installed_libraries: Dict[str, str] = Field(default_factory=dict)


class SystemInspector:
    """Discovers host hardware and installed software packages deterministically."""

    @staticmethod
    def inspect() -> HardwareCapabilities:
        """Probe CPU, RAM, CUDA, and installed Python packages without mutating state."""
        # 1. CPU and RAM
        cpu_count = os.cpu_count() or 1
        ram_gb = 8.0
        try:
            import psutil
            ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception:
            pass

        # 2. GPU Detection
        gpu_avail = False
        gpu_count = 0
        gpu_name = None
        try:
            import torch
            gpu_avail = torch.cuda.is_available()
            if gpu_avail:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

        # 3. Installed library versions
        target_pkgs = [
            "pandas", "polars", "numpy", "scipy", "sklearn",
            "xgboost", "lightgbm", "catboost", "torch", "mlflow", "pydantic"
        ]
        installed = {}
        for pkg in target_pkgs:
            try:
                m = __import__(pkg)
                installed[pkg] = getattr(m, "__version__", "available")
            except ImportError:
                installed[pkg] = "NOT_INSTALLED"

        return HardwareCapabilities(
            os_platform=platform.platform(),
            python_version=sys.version.split()[0],
            cpu_cores=cpu_count,
            system_ram_gb=ram_gb,
            gpu_available=gpu_avail,
            gpu_device_count=gpu_count,
            gpu_device_name=gpu_name,
            installed_libraries=installed,
        )
