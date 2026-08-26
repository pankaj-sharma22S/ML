"""Cross-encoder / feature-based neural reranker for context refinement."""

import re
from typing import Dict, List, Optional
from amea.rag.models import (
    DocumentChunk,
    RetrievalCandidate,
    SourceAuthority,
)


class CrossEncoderReranker:
    """Reranks candidate chunks based on token coverage, exact sequence matching, and authority."""

    AUTHORITY_WEIGHTS: Dict[SourceAuthority, float] = {
        SourceAuthority.OFFICIAL_POLICY: 1.25,
        SourceAuthority.INTERNAL_DOCUMENTATION: 1.0,
        SourceAuthority.TECHNICAL_NOTES: 0.85,
        SourceAuthority.UNVERIFIED_USER: 0.70,
    }

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        authorities_map: Optional[Dict[str, SourceAuthority]] = None,
    ) -> List[RetrievalCandidate]:
        """Rerank candidates and select top-k most relevant and authoritative chunks."""
        if not candidates:
            return []

        q_terms = set(re.findall(r"\b\w+\b", query.lower()))
        q_phrase = query.lower().strip()
        auth_map = authorities_map or {}

        for cand in candidates:
            content_lower = cand.chunk.content.lower()
            c_terms = set(re.findall(r"\b\w+\b", content_lower))

            # 1. Term coverage ratio
            coverage = len(q_terms.intersection(c_terms)) / len(q_terms) if q_terms else 0.0

            # 2. Exact phrase bonus
            phrase_bonus = 0.5 if q_phrase in content_lower else 0.0

            # 3. Source authority weight
            doc_id = cand.chunk.metadata.document_id
            authority = auth_map.get(doc_id, SourceAuthority.INTERNAL_DOCUMENTATION)
            auth_weight = self.AUTHORITY_WEIGHTS.get(authority, 1.0)

            # Combined reranking score
            base_score = (coverage * 0.5) + phrase_bonus + (cand.rrf_score * 10.0)
            cand.rerank_score = float(base_score * auth_weight)

        # Sort by rerank score descending
        sorted_reranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        return sorted_reranked[:self.top_k]
