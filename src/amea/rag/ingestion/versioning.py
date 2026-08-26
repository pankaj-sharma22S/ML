"""Document versioning, temporal lineage, and incremental update manager."""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
from amea.rag.models import (
    DocumentMetadata,
    DocumentRecord,
    DocumentVersion,
    SourceAuthority,
)


class DocumentVersionManager:
    """Maintains immutable document versions and temporal validity intervals."""

    def __init__(self):
        self.documents: Dict[str, DocumentRecord] = {} # doc_id -> DocumentRecord

    def register_new_document(
        self,
        document_id: str,
        source_id: str,
        filename: str,
        content_hash: str,
        authority: SourceAuthority = SourceAuthority.INTERNAL_DOCUMENTATION,
        tenant_id: str = "default_tenant",
        access_scope: Optional[List[str]] = None,
        effective_date: Optional[str] = None,
        chunks_count: int = 0,
    ) -> Tuple[DocumentRecord, DocumentVersion]:
        """Creates the initial canonical document record with Version 1."""
        now = datetime.now(timezone.utc)
        version_id = f"v1_{uuid4().hex[:6]}"

        v1 = DocumentVersion(
            version_id=version_id,
            document_id=document_id,
            version_number=1,
            content_hash=content_hash,
            is_current=True,
            valid_from=now,
            valid_to=None,
            created_at=now,
            chunks_count=chunks_count,
        )

        metadata = DocumentMetadata(
            document_id=document_id,
            source_id=source_id,
            filename=filename,
            authority=authority,
            tenant_id=tenant_id,
            access_scope=access_scope or ["public"],
            effective_date=effective_date,
            file_hash=content_hash,
            normalized_content_hash=content_hash,
            created_at=now,
            updated_at=now,
        )

        record = DocumentRecord(
            document_id=document_id,
            metadata=metadata,
            versions=[v1],
            current_version_id=version_id,
        )

        self.documents[document_id] = record
        return record, v1

    def add_new_version(
        self,
        document_id: str,
        content_hash: str,
        effective_date: Optional[str] = None,
        chunks_count: int = 0,
    ) -> Tuple[DocumentRecord, DocumentVersion]:
        """Supersedes previous current version with an updated immutable version."""
        if document_id not in self.documents:
            raise KeyError(f"Document {document_id} not found in version registry.")

        record = self.documents[document_id]
        now = datetime.now(timezone.utc)

        # Mark previous versions as not current and set valid_to timestamp
        for v in record.versions:
            if v.is_current:
                v.is_current = False
                v.valid_to = now

        new_version_num = len(record.versions) + 1
        new_version_id = f"v{new_version_num}_{uuid4().hex[:6]}"

        new_version = DocumentVersion(
            version_id=new_version_id,
            document_id=document_id,
            version_number=new_version_num,
            content_hash=content_hash,
            is_current=True,
            valid_from=now,
            valid_to=None,
            created_at=now,
            chunks_count=chunks_count,
        )

        record.versions.append(new_version)
        record.current_version_id = new_version_id
        record.metadata.file_hash = content_hash
        record.metadata.normalized_content_hash = content_hash
        record.metadata.updated_at = now
        if effective_date:
            record.metadata.effective_date = effective_date

        return record, new_version

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        return self.documents.get(document_id)

    def get_version(self, document_id: str, version_id: str) -> Optional[DocumentVersion]:
        doc = self.documents.get(document_id)
        if not doc:
            return None
        for v in doc.versions:
            if v.version_id == version_id:
                return v
        return None
