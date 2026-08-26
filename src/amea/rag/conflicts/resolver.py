"""Conflict resolution engine using source authority, temporal recency, and human escalation."""

from typing import List, Tuple
from amea.rag.models import (
    ConflictRecord,
    SourceAuthority,
)


class ConflictResolver:
    """Resolves cross-document conflicts or escalates to human review."""

    AUTHORITY_RANKS = {
        SourceAuthority.OFFICIAL_POLICY: 4,
        SourceAuthority.INTERNAL_DOCUMENTATION: 3,
        SourceAuthority.TECHNICAL_NOTES: 2,
        SourceAuthority.UNVERIFIED_USER: 1,
    }

    @classmethod
    def resolve_conflicts(cls, conflicts: List[ConflictRecord]) -> Tuple[List[ConflictRecord], List[ConflictRecord]]:
        """
        Attempts resolution of detected conflicts.
        Returns (resolved_conflicts, unresolved_conflicts).
        """
        resolved: List[ConflictRecord] = []
        unresolved: List[ConflictRecord] = []

        for conf in conflicts:
            rank_a = cls.AUTHORITY_RANKS.get(conf.authority_a, 1)
            rank_b = cls.AUTHORITY_RANKS.get(conf.authority_b, 1)

            # 1. Authority Hierarchy Resolution
            if rank_a > rank_b:
                conf.is_resolved = True
                conf.superseding_source = conf.source_a
                conf.resolution_summary = (
                    f"Resolved via Source Authority: '{conf.source_a}' ({conf.authority_a.value}) "
                    f"supersedes '{conf.source_b}' ({conf.authority_b.value})."
                )
                resolved.append(conf)
            elif rank_b > rank_a:
                conf.is_resolved = True
                conf.superseding_source = conf.source_b
                conf.resolution_summary = (
                    f"Resolved via Source Authority: '{conf.source_b}' ({conf.authority_b.value}) "
                    f"supersedes '{conf.source_a}' ({conf.authority_a.value})."
                )
                resolved.append(conf)
            else:
                # 2. Temporal Recency Resolution (if same authority)
                # Check for year indicators in claims
                year_a = 2024 if "2024" in conf.claim_a else (2025 if "2025" in conf.claim_a else None)
                year_b = 2024 if "2024" in conf.claim_b else (2025 if "2025" in conf.claim_b else None)

                if year_a and year_b and year_a != year_b:
                    newer_source = conf.source_a if year_a > year_b else conf.source_b
                    older_source = conf.source_b if year_a > year_b else conf.source_a
                    conf.is_resolved = True
                    conf.superseding_source = newer_source
                    conf.resolution_summary = (
                        f"Resolved via Temporal Recency: '{newer_source}' is the newer version and supersedes '{older_source}'."
                    )
                    resolved.append(conf)
                else:
                    # Equal authority and cannot be determined -> Escalate
                    conf.is_resolved = False
                    conf.requires_human_clarification = True
                    conf.resolution_summary = (
                        f"Unresolvable conflict between '{conf.source_a}' and '{conf.source_b}' with equal authority. "
                        f"Requires human clarification."
                    )
                    unresolved.append(conf)

        return resolved, unresolved
