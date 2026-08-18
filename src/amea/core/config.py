"""System configuration and user-authorized resource budgets."""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class ExecutionLimits(BaseModel):
    """Execution sandbox resource limits."""
    max_timeout_seconds: int = Field(default=300, description="Per-experiment timeout in seconds")
    max_memory_mb: int = Field(default=4096, description="Max RAM allowed per worker in MB")
    max_processes: int = Field(default=4, description="Max concurrent worker subprocesses")
    allow_network: bool = Field(default=False, description="Whether workers can make external network calls")


class SecurityConfig(BaseModel):
    """Security constraints for untrusted execution."""
    sandbox_root: Path = Field(default=Path(".amea_sandboxes"), description="Root directory for worker sandboxes")
    strip_environment_secrets: bool = Field(default=True, description="Remove API keys and secrets from worker env")
    blocked_env_patterns: list[str] = Field(
        default_factory=lambda: ["*API_KEY*", "*SECRET*", "*TOKEN*", "*PASSWORD*", "*CREDENTIALS*"]
    )
    blocked_commands: list[str] = Field(
        default_factory=lambda: ["rm -rf /", "shutdown", "reboot", "format", "mkfs"]
    )


class ComputeBudget(BaseModel):
    """User-authorized budget boundaries (strictly separated from detected hardware capabilities)."""
    max_experiments: int = Field(default=10, description="Total maximum experiments allowed")
    max_total_time_seconds: int = Field(default=3600, description="Global wallclock time limit")
    max_improvement_iterations: int = Field(default=3, description="Maximum self-improvement loops")
    max_cpu_cores: int = Field(default=4, description="User-authorized CPU cores for workers")
    max_gpu_count: int = Field(default=0, description="User-authorized GPUs")
    max_cost_dollars: Optional[float] = Field(default=None, description="Optional monetary cap")


class PersistenceConfig(BaseModel):
    """Storage and checkpointing settings."""
    project_dir: Path = Field(default=Path(".amea_project"), description="Root persistence directory")
    checkpoint_interval_steps: int = Field(default=1, description="Persist state every N steps")
    enable_wal: bool = Field(default=True, description="Enable Write-Ahead-Log for persistence")


class ProjectConfig(BaseModel):
    """Global configuration for an AMEA run."""
    project_id: str = Field(default="amea-default", description="Unique project ID")
    workspace_root: Path = Field(default=Path("."), description="User workspace root")
    budget: ComputeBudget = Field(default_factory=ComputeBudget)
    limits: ExecutionLimits = Field(default_factory=ExecutionLimits)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    headless: bool = Field(default=False, description="Run without interactive confirmation prompts")
