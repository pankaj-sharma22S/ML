"""Environment and secret management for AMEA."""

import os
from pathlib import Path
from typing import Optional
from dotenv import find_dotenv, load_dotenv

# Search for root .env and load into os.environ
_ENV_PATH = find_dotenv(usecwd=True)
if _ENV_PATH:
    load_dotenv(_ENV_PATH)
else:
    # Try looking in parent directories
    fallback_env = Path(__file__).resolve().parents[3] / ".env"
    if fallback_env.exists():
        load_dotenv(str(fallback_env))


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve environment variable safely."""
    return os.environ.get(key, default)


def mask_secret(secret: Optional[str], show_chars: int = 4) -> str:
    """Mask sensitive tokens and API keys for display in UI or logs."""
    if not secret or len(secret) < 6:
        return "********" if secret else "NOT_CONFIGURED"
    if len(secret) <= (show_chars * 2):
        return "****"
    return f"{secret[:show_chars]}...{secret[-show_chars:]}"


def scrub_secrets_from_text(text: str) -> str:
    """Redact known environment secrets from stdout/stderr/log strings."""
    if not text:
        return ""
    sensitive_keys = [
        "OPENROUTER_API_KEY",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_JWT_SECRET",
        "AWS_SECRET_ACCESS_KEY",
        "SECRET_KEY",
    ]
    scrubbed = text
    for key in sensitive_keys:
        val = os.environ.get(key)
        if val and len(val) >= 6 and val in scrubbed:
            scrubbed = scrubbed.replace(val, f"[REDACTED_SECRET_{key}]")
    return scrubbed
