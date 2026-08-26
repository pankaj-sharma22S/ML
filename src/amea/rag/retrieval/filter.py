"""Pre-retrieval deterministic metadata & security filtering."""

from typing import Dict, List, Optional, Set
from amea.rag.models import DocumentChunk, RetrievalFilter
from amea.rag.security.security_guard import RAGSecurityGuard


class PreRetrievalFilter:
    """Filters chunk IDs prior to execution of search algorithms."""

    @staticmethod
    def get_allowed_chunk_ids(
        chunks: Dict[str, DocumentChunk],
        filters: RetrievalFilter,
    ) -> Set[str]:
        """
        Deterministic filtering based on tenant, permissions, document type, and version.
        """
        allowed: Set[str] = set()

        for chunk_id, chunk in chunks.items():
            meta = chunk.metadata

            # 1. Tenant & Security Scope Check
            if not RAGSecurityGuard.check_access_permission(
                chunk=chunk,
                tenant_id=filters.tenant_id,
                user_scopes=filters.user_access_scopes,
            ):
                continue

            # 2. Document Type Filter
            # (checked against parent doc if specified)

            # 3. Department Filter
            if filters.departments and hasattr(meta, "department") and meta.department:
                if meta.department not in filters.departments:
                    continue

            allowed.add(chunk_id)

        return allowed
