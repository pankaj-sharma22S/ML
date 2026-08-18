"""Security boundaries and enforcement for untrusted code execution."""

import fnmatch
import os
from pathlib import Path
from typing import Dict, List
from amea.core.exceptions import SecurityViolationError


class SecurityBoundary:
    """Enforces sandbox constraints, environment scrubbing, and command validation."""

    def __init__(
        self,
        allowed_root: Path,
        blocked_env_patterns: List[str] | None = None,
        blocked_command_substrings: List[str] | None = None,
    ):
        self.allowed_root = allowed_root.resolve()
        self.blocked_env_patterns = blocked_env_patterns or [
            "*API_KEY*", "*SECRET*", "*TOKEN*", "*PASSWORD*", "*CREDENTIAL*", "*AUTH*"
        ]
        self.blocked_command_substrings = blocked_command_substrings or [
            "rm -rf /", "mkfs", "shutdown", "reboot", "format c:", "drop database", "powershell -enc"
        ]

    def validate_path(self, path: Path | str) -> Path:
        """Ensure path is within the allowed workspace boundary."""
        resolved = Path(path).resolve()
        if not (str(resolved) == str(self.allowed_root) or str(resolved).startswith(str(self.allowed_root) + os.sep)):
            raise SecurityViolationError(f"Security violation: Path '{resolved}' escapes allowed boundary '{self.allowed_root}'.")
        return resolved

    def validate_command(self, command_str: str) -> None:
        """Inspect command string for dangerous shell commands or scripts."""
        cmd_lower = command_str.lower()
        for blocked in self.blocked_command_substrings:
            if blocked.lower() in cmd_lower:
                raise SecurityViolationError(f"Security violation: Blocked command pattern detected: '{blocked}'")

    def sanitize_environment(self, base_env: Dict[str, str] | None = None) -> Dict[str, str]:
        """Strip sensitive secrets and credentials from subprocess environment."""
        source_env = os.environ.copy() if base_env is None else base_env
        clean_env = {}
        for key, value in source_env.items():
            key_upper = key.upper()
            if any(fnmatch.fnmatch(key_upper, pat) for pat in self.blocked_env_patterns):
                continue  # Scrub secret
            clean_env[key] = value
        # Ensure minimal necessary system paths exist
        clean_env["PYTHONUNBUFFERED"] = "1"
        clean_env["AMEA_SANDBOX_ACTIVE"] = "1"
        return clean_env
