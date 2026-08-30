"""Security boundaries, AST validation, dependency validation, and environment scrubbing."""

import ast
import fnmatch
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from amea.core.exceptions import SecurityViolationError


class AstSecurityValidator:
    """Inspects Python source code AST to disallow dangerous operations before execution."""

    # Forbidden module imports
    FORBIDDEN_MODULES: Set[str] = {
        "subprocess",
        "socket",
        "socketserver",
        "http",
        "urllib",
        "requests",
        "httpx",
        "ftplib",
        "telnetlib",
        "smtplib",
        "paramiko",
        "pyautogui",
        "pynput",
        "keyring",
        "ctypes",
        "winreg",
        "pty",
        "posix",
    }

    # Forbidden dangerous builtins / function calls
    FORBIDDEN_CALLS: Set[str] = {
        "eval",
        "exec",
        "__import__",
        "compile",
        "globals",
        "locals",
        "breakpoint",
    }

    # Forbidden attribute access patterns
    FORBIDDEN_ATTRIBUTES: Set[str] = {
        "system",
        "popen",
        "spawn",
        "fork",
        "kill",
        "remove",
        "rmdir",
        "unlink",
        "chmod",
        "chown",
        "rmtree",
        "move",
    }

    # Forbidden sensitive path indicators in string literals
    SENSITIVE_PATH_PATTERNS: List[str] = [
        r"\.ssh",
        r"\.aws",
        r"\.kube",
        r"/etc/passwd",
        r"/etc/shadow",
        r"C:\\Windows\\System32",
        r"\.env",
        r"id_rsa",
    ]

    @classmethod
    def validate_code_safety(cls, code_str: str) -> List[str]:
        """
        Parses and inspects code AST. Returns list of security violations found.
        Empty list indicates code is safe.
        """
        violations: List[str] = []

        try:
            tree = ast.parse(code_str)
        except SyntaxError:
            # Let the real Python kernel execute the syntax error to produce authentic traceback
            return []

        for node in ast.walk(tree):
            # 1. Inspect import statements
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod in cls.FORBIDDEN_MODULES:
                        violations.append(f"Forbidden module import detected: '{alias.name}' (line {node.lineno})")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    if root_mod in cls.FORBIDDEN_MODULES:
                        violations.append(f"Forbidden module import detected: '{node.module}' (line {node.lineno})")

            # 2. Inspect function calls
            elif isinstance(node, ast.Call):
                # Check direct function calls (e.g. eval(), exec())
                if isinstance(node.func, ast.Name):
                    if node.func.id in cls.FORBIDDEN_CALLS:
                        violations.append(f"Forbidden function call detected: '{node.func.id}()' (line {node.lineno})")

                # Check attribute calls (e.g. os.system(), subprocess.run())
                elif isinstance(node.func, ast.Attribute):
                    attr_name = node.func.attr
                    if attr_name in cls.FORBIDDEN_ATTRIBUTES:
                        # If called on os, posix, sys, shutil
                        if isinstance(node.func.value, ast.Name):
                            mod_name = node.func.value.id
                            if mod_name in ["os", "posix", "sys", "shutil"]:
                                violations.append(f"Dangerous system call detected: '{mod_name}.{attr_name}()' (line {node.lineno})")

            # 3. Inspect string literals for sensitive path traversal / credentials
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                for pattern in cls.SENSITIVE_PATH_PATTERNS:
                    if re.search(pattern, node.value, re.IGNORECASE):
                        violations.append(f"Sensitive path reference detected: '{node.value}' (line {node.lineno})")

        return violations


class DependencySecurityValidator:
    """Validates third-party package dependencies against an enterprise allowlist."""

    APPROVED_PACKAGES: Set[str] = {
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
    }

    BLOCKED_PACKAGES: Set[str] = {
        "requests",
        "urllib3",
        "httpx",
        "paramiko",
        "pyautogui",
        "pynput",
        "keyring",
        "cryptography",
        "scapy",
        "fabric",
        "ansible",
    }

    @classmethod
    def validate_dependencies(cls, requirements: List[str]) -> Tuple[bool, List[str]]:
        """
        Validates dependency manifest.
        Returns (is_valid, list_of_violations).
        """
        violations: List[str] = []

        for req in requirements:
            req_clean = req.strip()
            if not req_clean or req_clean.startswith("#"):
                continue

            # Extract base package name
            pkg_name = re.split(r"[><=~!;]", req_clean)[0].strip().lower()

            if pkg_name in cls.BLOCKED_PACKAGES:
                violations.append(f"Package '{pkg_name}' is explicitly BLOCKED due to security policy.")
            elif pkg_name not in cls.APPROVED_PACKAGES:
                violations.append(f"Package '{pkg_name}' is NOT in approved ML package allowlist.")

        return (len(violations) == 0), violations


class EnvironmentSanitizer:
    """Two-sided environment sanitizer protecting host credentials and secrets."""

    BLOCKED_ENV_PATTERNS: List[str] = [
        "*API_KEY*",
        "*SECRET*",
        "*TOKEN*",
        "*PASSWORD*",
        "*CREDENTIAL*",
        "*AUTH*",
        "*AWS_*",
        "*GITHUB_*",
        "*SSH_*",
        "*OPENAI_*",
        "*GEMINI_*",
        "*ANTHROPIC_*",
        "*PRIVATE*",
    ]

    @classmethod
    def sanitize(cls, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Strip sensitive secrets and credentials from subprocess environment."""
        source_env = os.environ.copy() if base_env is None else base_env
        clean_env = {}

        for key, value in source_env.items():
            key_upper = key.upper()
            if any(fnmatch.fnmatch(key_upper, pat) for pat in cls.BLOCKED_ENV_PATTERNS):
                continue  # Scrub secret
            clean_env[key] = value

        # Ensure minimal necessary execution variables
        clean_env["PYTHONUNBUFFERED"] = "1"
        clean_env["AMEA_SANDBOX_ACTIVE"] = "1"
        clean_env["MPLBACKEND"] = "Agg"
        return clean_env


class SecurityBoundary:
    """Enforces sandbox constraints, path boundaries, AST validation, and environment scrubbing."""

    def __init__(
        self,
        allowed_root: Path,
        blocked_env_patterns: Optional[List[str]] = None,
        blocked_command_substrings: Optional[List[str]] = None,
    ):
        self.allowed_root = allowed_root.resolve()
        self.blocked_env_patterns = blocked_env_patterns or EnvironmentSanitizer.BLOCKED_ENV_PATTERNS
        self.blocked_command_substrings = blocked_command_substrings or [
            "rm -rf /", "mkfs", "shutdown", "reboot", "format c:", "drop database", "powershell -enc"
        ]
        self.ast_validator = AstSecurityValidator()
        self.dep_validator = DependencySecurityValidator()
        self.env_sanitizer = EnvironmentSanitizer()

    def validate_path(self, path: Path | str) -> Path:
        """Ensure path is within the allowed workspace boundary."""
        resolved = Path(path).resolve()
        if not (str(resolved) == str(self.allowed_root) or str(resolved).startswith(str(self.allowed_root) + os.sep)):
            raise SecurityViolationError(f"Security violation: Path '{resolved}' escapes allowed boundary '{self.allowed_root}'.")
        return resolved

    def validate_code(self, code_str: str) -> None:
        """Inspect code AST for dangerous operations."""
        violations = self.ast_validator.validate_code_safety(code_str)
        if violations:
            raise SecurityViolationError(f"Code security violation: {'; '.join(violations)}")

    def validate_dependencies(self, requirements: List[str]) -> None:
        """Inspect dependency manifest."""
        is_valid, violations = self.dep_validator.validate_dependencies(requirements)
        if not is_valid:
            raise SecurityViolationError(f"Dependency security violation: {'; '.join(violations)}")

    def validate_command(self, command_str: str) -> None:
        """Inspect command string for dangerous shell commands or scripts."""
        cmd_lower = command_str.lower()
        for blocked in self.blocked_command_substrings:
            if blocked.lower() in cmd_lower:
                raise SecurityViolationError(f"Security violation: Blocked command pattern detected: '{blocked}'")

    def sanitize_environment(self, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Strip sensitive secrets and credentials from subprocess environment."""
        return self.env_sanitizer.sanitize(base_env)
