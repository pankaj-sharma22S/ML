"""Document Ingestion, Deduplication, Chunking & Versioning package."""

from amea.rag.ingestion.parsers import DocumentParser
from amea.rag.ingestion.deduplicator import DocumentDeduplicator, DuplicateStatus
from amea.rag.ingestion.chunker import HierarchicalChunker
from amea.rag.ingestion.versioning import DocumentVersionManager

__all__ = [
    "DocumentParser",
    "DocumentDeduplicator",
    "DuplicateStatus",
    "HierarchicalChunker",
    "DocumentVersionManager",
]
