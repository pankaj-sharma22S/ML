"""Conflict detector for identifying discrepancies across multi-document corpora."""

import re
from typing import Dict, List, Optional
from uuid import uuid4
from amea.rag.models import (
    ConflictRecord,
    ConflictSeverity,
    ConflictType,
    DocumentChunk,
    SourceAuthority,
)


class MultiDocumentConflictDetector:
    """Detects temporal, numerical, semantic, and source authority conflicts across evidence."""

    @classmethod
    def detect_conflicts(
        cls,
        chunks: List[DocumentChunk],
        authorities: Dict[str, SourceAuthority],
    ) -> List[ConflictRecord]:
        """Scan candidate chunks for cross-document factual contradictions."""
        conflicts: List[ConflictRecord] = []

        # Simple semantic/numerical pattern scanner
        # Pattern 1: Numerical claims regarding key topics (e.g. leave days, revenue, timeout)
        numerical_topics = ["leave", "vacation", "revenue", "timeout", "retention", "days", "budget"]

        topic_claims: Dict[str, List[Dict]] = {} # topic -> list of {doc_id, value, sentence, chunk, authority}

        for chunk in chunks:
            doc_id = chunk.metadata.document_id
            auth = authorities.get(doc_id, SourceAuthority.INTERNAL_DOCUMENTATION)
            sentences = re.split(r"[.\n]", chunk.content)

            for sent in sentences:
                sent_lower = sent.lower()
                for topic in numerical_topics:
                    if topic in sent_lower:
                        # Extract any numbers associated with topic
                        nums = re.findall(r"\$?\b\d+(?:\.\d+)?(?:\s*(?:days|million|seconds|ms|k|m))?\b", sent_lower)
                        if nums:
                            if topic not in topic_claims:
                                topic_claims[topic] = []
                            topic_claims[topic].append({
                                "doc_id": doc_id,
                                "claim": sent.strip(),
                                "num_val": nums[0],
                                "authority": auth,
                                "date": chunk.metadata.section, # section or heading often contains year
                                "chunk": chunk,
                            })

        # Compare claims per topic
        for topic, claims in topic_claims.items():
            if len(claims) >= 2:
                for i in range(len(claims)):
                    for j in range(i + 1, len(claims)):
                        c1, c2 = claims[i], claims[j]
                        if c1["doc_id"] != c2["doc_id"] and c1["num_val"] != c2["num_val"]:
                            # Detected conflict
                            conf_type = ConflictType.NUMERICAL
                            if any(yr in c1["claim"] or yr in c2["claim"] for yr in ["2024", "2025", "2026"]):
                                conf_type = ConflictType.TEMPORAL

                            conflicts.append(ConflictRecord(
                                conflict_id=f"conf_{uuid4().hex[:6]}",
                                conflict_type=conf_type,
                                severity=ConflictSeverity.CRITICAL if conf_type == ConflictType.TEMPORAL else ConflictSeverity.IMPORTANT,
                                topic_or_entity=topic,
                                source_a=c1["doc_id"],
                                claim_a=c1["claim"],
                                authority_a=c1["authority"],
                                date_a=str(c1["date"]),
                                source_b=c2["doc_id"],
                                claim_b=c2["claim"],
                                authority_b=c2["authority"],
                                date_b=str(c2["date"]),
                                evidence=f"Source '{c1['doc_id']}' claims '{c1['num_val']}' while Source '{c2['doc_id']}' claims '{c2['num_val']}' for '{topic}'.",
                            ))

        return conflicts
