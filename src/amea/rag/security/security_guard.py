"""Security and Prompt-Injection Guard protecting the retrieval and LLM context."""

import re
from typing import List, Tuple
from amea.rag.models import (
    DocumentChunk,
    DocumentMetadata,
    InjectionRisk,
    SecurityStatus,
)


class RAGSecurityGuard:
    """Detects adversarial prompt injections and enforces tenant security boundaries."""

    # Adversarial prompt injection patterns
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"reveal\s+(the\s+)?(system\s+prompt|developer\s+mode|secret\s+key|password)",
        r"disregard\s+(the\s+)?(safety\s+rules|system\s+directives)",
        r"override\s+(all\s+)?(instructions|policies|prompts)",
        r"you\s+are\s+now\s+(in\s+dan\s+mode|unrestricted|a\s+different\s+ai)",
        r"exfiltrate\s+(data|tokens|keys|environment\s+variables)",
    ]

    @classmethod
    def audit_text_security(cls, text: str) -> Tuple[InjectionRisk, SecurityStatus, List[str]]:
        """Scans content for prompt injection triggers."""
        matched_flags = []
        lower = text.lower()

        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, lower, re.IGNORECASE):
                matched_flags.append(f"Prompt injection trigger detected: '{pattern}'")

        if len(matched_flags) >= 2:
            return InjectionRisk.CRITICAL, SecurityStatus.REJECTED, matched_flags
        elif len(matched_flags) == 1:
            return InjectionRisk.MEDIUM, SecurityStatus.QUARANTINED, matched_flags

        return InjectionRisk.NONE, SecurityStatus.TRUSTED, []

    @classmethod
    def sanitize_retrieved_chunk_for_llm(cls, chunk: DocumentChunk) -> str:
        """
        Wraps retrieved chunk content in passive data delimiters to ensure
        the LLM treats the text purely as inert reference DATA.
        """
        escaped_content = chunk.content.replace("<SYSTEM_DIRECTIVE>", "[REDACTED]")
        return (
            f"<RETRIEVED_DOCUMENT_DATA chunk_id=\"{chunk.chunk_id}\" source=\"{chunk.metadata.source_uri}\">\n"
            f"{escaped_content}\n"
            f"</RETRIEVED_DOCUMENT_DATA>"
        )

    @classmethod
    def check_access_permission(
        cls,
        chunk: DocumentChunk,
        tenant_id: str,
        user_scopes: List[str],
    ) -> bool:
        """Deterministic tenant and scope authorization check."""
        if chunk.metadata.tenant_id != tenant_id:
            return False

        chunk_scopes = set(chunk.metadata.access_scope)
        user_scope_set = set(user_scopes)

        # "public" is accessible by all; otherwise user must hold at least one matching scope
        if "public" in chunk_scopes or not chunk_scopes.isdisjoint(user_scope_set):
            return True

        return False
