"""Artifact manager for persisting visualization charts and analysis outputs."""

from pathlib import Path
from typing import Optional
from uuid import uuid4
import matplotlib.pyplot as plt


class AnalysisArtifactManager:
    """Manages creation and storage of visual artifacts outside source repository."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = (base_dir or Path(".amea_project/analysis_artifacts")).resolve()

    def create_run_directory(self, run_id: str) -> Path:
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_figure(self, fig: plt.Figure, run_id: str, name: str) -> str:
        run_dir = self.create_run_directory(run_id)
        chart_id = f"{name}_{uuid4().hex[:6]}.png"
        target_path = run_dir / chart_id
        fig.savefig(target_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return str(target_path)
