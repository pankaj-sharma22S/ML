"""Structured checkpoint persistence with schema versioning and recovery."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

from amea.core.exceptions import CheckpointRecoveryError
from amea.core.state import GlobalState


class CheckpointMetadata(BaseModel):
    """Metadata describing a persisted state checkpoint."""
    checkpoint_id: str
    project_id: str
    task_id: str
    phase: str
    iteration: int
    schema_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    file_name: str


class StateCheckpointer:
    """Manages persistent snapshots and WAL for GlobalState."""

    def __init__(self, persistence_dir: Path):
        self.persistence_dir = persistence_dir.resolve()
        self.checkpoints_dir = self.persistence_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.checkpoints_dir / "index.json"

    def save_checkpoint(self, state: GlobalState, checkpoint_id: Optional[str] = None) -> CheckpointMetadata:
        """Persist a GlobalState snapshot with metadata."""
        cp_id = checkpoint_id or f"cp_{state.current_phase}_{int(datetime.now(timezone.utc).timestamp())}"
        file_name = f"{cp_id}.json"
        target_path = self.checkpoints_dir / file_name

        data = {
            "metadata": {
                "checkpoint_id": cp_id,
                "project_id": state.project_id,
                "task_id": state.task_id,
                "phase": state.current_phase.value,
                "iteration": state.iteration,
                "schema_version": state.schema_version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "file_name": file_name,
            },
            "state": state.model_dump(mode="json"),
        }

        target_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        meta = CheckpointMetadata.model_validate(data["metadata"])
        self._update_index(meta)
        return meta

    def load_checkpoint(self, checkpoint_id: str) -> GlobalState:
        """Load and deserialize a specific checkpoint."""
        target_path = self.checkpoints_dir / f"{checkpoint_id}.json"
        if not target_path.exists():
            raise CheckpointRecoveryError(f"Checkpoint '{checkpoint_id}' not found at {target_path}")

        try:
            content = json.loads(target_path.read_text(encoding="utf-8"))
            return GlobalState.model_validate(content["state"])
        except Exception as e:
            raise CheckpointRecoveryError(f"Failed to recover state from checkpoint '{checkpoint_id}': {e}") from e

    def load_latest_checkpoint(self) -> Optional[GlobalState]:
        """Load the most recent valid checkpoint if available."""
        index = self.list_checkpoints()
        if not index:
            return None
        latest = index[-1]
        return self.load_checkpoint(latest.checkpoint_id)

    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints ordered chronologically."""
        if not self.index_file.exists():
            return []
        try:
            raw = json.loads(self.index_file.read_text(encoding="utf-8"))
            return [CheckpointMetadata.model_validate(m) for m in raw]
        except Exception:
            return []

    def _update_index(self, meta: CheckpointMetadata) -> None:
        """Append metadata record to index."""
        current = self.list_checkpoints()
        current.append(meta)
        self.index_file.write_text(
            json.dumps([m.model_dump(mode="json") for m in current], indent=2),
            encoding="utf-8",
        )
