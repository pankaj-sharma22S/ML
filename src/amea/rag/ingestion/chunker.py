"""Structure-aware hierarchical parent-child chunker."""

import hashlib
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4
from amea.rag.models import ChunkMetadata, DocumentChunk


class HierarchicalChunker:
    """Splits structured sections into parent sections and child search chunks."""

    def __init__(
        self,
        child_chunk_size: int = 300, # words
        child_chunk_overlap: int = 50, # words
    ):
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap

    def chunk_document(
        self,
        document_id: str,
        version_id: str,
        source_uri: str,
        structured_sections: List[Dict[str, Any]],
        tenant_id: str = "default_tenant",
        access_scope: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        """
        Produce searchable child chunks linked to their parent section.
        """
        scopes = access_scope or ["public"]
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for sec_idx, section in enumerate(structured_sections):
            heading = section.get("heading", f"Section {sec_idx + 1}")
            sec_content = section.get("content", "")
            page = section.get("page", 1)

            if not sec_content.strip():
                continue

            parent_chunk_id = f"parent_{document_id}_v{version_id}_s{sec_idx}"
            words = sec_content.split()

            # If section is small enough, it is its own child chunk
            if len(words) <= self.child_chunk_size:
                c_hash = hashlib.sha256(sec_content.encode("utf-8")).hexdigest()
                meta = ChunkMetadata(
                    chunk_id=f"chunk_{document_id}_{global_chunk_idx}",
                    document_id=document_id,
                    version_id=version_id,
                    chunk_index=global_chunk_idx,
                    page=page,
                    section=heading,
                    heading=heading,
                    parent_chunk_id=parent_chunk_id,
                    source_uri=source_uri,
                    tenant_id=tenant_id,
                    access_scope=scopes,
                    content_hash=c_hash,
                    token_count=len(words),
                )
                chunk = DocumentChunk(
                    chunk_id=meta.chunk_id,
                    content=sec_content,
                    metadata=meta,
                    parent_content=sec_content,
                )
                all_chunks.append(chunk)
                global_chunk_idx += 1
            else:
                # Sliding window of child chunks with overlap
                start = 0
                while start < len(words):
                    end = min(start + self.child_chunk_size, len(words))
                    child_words = words[start:end]
                    child_text = " ".join(child_words)
                    c_hash = hashlib.sha256(child_text.encode("utf-8")).hexdigest()

                    meta = ChunkMetadata(
                        chunk_id=f"chunk_{document_id}_{global_chunk_idx}",
                        document_id=document_id,
                        version_id=version_id,
                        chunk_index=global_chunk_idx,
                        page=page,
                        section=heading,
                        heading=heading,
                        parent_chunk_id=parent_chunk_id,
                        source_uri=source_uri,
                        tenant_id=tenant_id,
                        access_scope=scopes,
                        content_hash=c_hash,
                        token_count=len(child_words),
                    )
                    chunk = DocumentChunk(
                        chunk_id=meta.chunk_id,
                        content=child_text,
                        metadata=meta,
                        parent_content=sec_content, # Expanded parent context
                    )
                    all_chunks.append(chunk)
                    global_chunk_idx += 1

                    if end == len(words):
                        break
                    start += (self.child_chunk_size - self.child_chunk_overlap)

        return all_chunks
