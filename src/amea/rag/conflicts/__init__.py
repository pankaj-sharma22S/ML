"""Conflict Detection & Resolution package."""

from amea.rag.conflicts.detector import MultiDocumentConflictDetector
from amea.rag.conflicts.resolver import ConflictResolver

__all__ = [
    "MultiDocumentConflictDetector",
    "ConflictResolver",
]
