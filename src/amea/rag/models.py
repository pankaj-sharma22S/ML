"""Comprehensive Pydantic v2 schemas and shared state contracts for the Enterprise RAG Platform."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"
    PDF = "pdf"
    DOCX = "docx"
    SOURCE_CODE = "source_code"
    UNSPECIFIED = "unspecified"


class SourceAuthority(str, Enum):
    OFFICIAL_POLICY = "OFFICIAL_POLICY"              # Highest authority (e.g. current approved company policy)
    INTERNAL_DOCUMENTATION = "INTERNAL_DOCUMENTATION" # High authority (e.g. system architecture, verified docs)
    TECHNICAL_NOTES = "TECHNICAL_NOTES"              # Medium authority (e.g. developer scratchpad, discussion)
    UNVERIFIED_USER = "UNVERIFIED_USER"              # Low authority (e.g. unvetted upload)


class SecurityStatus(str, Enum):
    TRUSTED = "TRUSTED"
    QUARANTINED = "QUARANTINED"
    REJECTED = "REJECTED"


class InjectionRisk(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    CRITICAL = "CRITICAL"


class ConflictType(str, Enum):
    TEMPORAL = "TEMPORAL"                # Version/date discrepancy (e.g. 2024 vs 2025 leave days)
    SOURCE_AUTHORITY = "SOURCE_AUTHORITY" # Conflicting claims between different authority levels
    NUMERICAL = "NUMERICAL"              # Conflicting numbers/metrics across reports
    SEMANTIC = "SEMANTIC"                # Opposite factual assertions (e.g. supported vs deprecated)
    ENTITY = "ENTITY"                    # Divergent definitions of the same entity


class ConflictSeverity(str, Enum):
    CRITICAL = "CRITICAL"    # Must resolve or escalate before answering
    IMPORTANT = "IMPORTANT"  # Should present both or resolve via authority
    MINOR = "MINOR"          # Documented discrepancy


class RelationshipType(str, Enum):
    STRUCTURAL = "STRUCTURAL"      # Foreign-key or schema match
    SEMANTIC = "SEMANTIC"          # Shared entity or topic
    TEMPORAL = "TEMPORAL"          # Version progression (supersedes/prior version)
    DOCUMENT_REF = "DOCUMENT_REF"  # Explicit citation or attachment
    ENTITY_MEMBER = "ENTITY_MEMBER" # Belonging to department/project


class ChunkMetadata(BaseModel):
    """Rich metadata attached to each searchable text chunk."""
    chunk_id: str
    document_id: str
    version_id: str
    chunk_index: int
    page: Optional[int] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    heading: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    source_uri: str
    tenant_id: str = "default_tenant"
    access_scope: List[str] = Field(default_factory=lambda: ["public"])
    language: str = "en"
    content_hash: str
    token_count: int = 0
    embedding_model: str = "text-embedding-3-small"
    embedding_version: str = "v1"


class DocumentChunk(BaseModel):
    """Searchable chunk unit containing content and metadata."""
    chunk_id: str
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    parent_content: Optional[str] = None  # Expanded context for parent-child retrieval


class DocumentMetadata(BaseModel):
    """Metadata describing a full source document."""
    document_id: str
    source_id: str
    filename: str
    document_type: DocumentType = DocumentType.TEXT
    author: Optional[str] = None
    owner: Optional[str] = None
    department: Optional[str] = None
    tenant_id: str = "default_tenant"
    access_scope: List[str] = Field(default_factory=lambda: ["public"])
    authority: SourceAuthority = SourceAuthority.INTERNAL_DOCUMENTATION
    effective_date: Optional[str] = None
    expiration_date: Optional[str] = None
    file_hash: str
    normalized_content_hash: str
    security_status: SecurityStatus = SecurityStatus.TRUSTED
    injection_risk: InjectionRisk = InjectionRisk.NONE
    security_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentVersion(BaseModel):
    """Represents a specific immutable version of a document."""
    version_id: str
    document_id: str
    version_number: int
    content_hash: str
    is_current: bool = True
    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    chunks_count: int = 0


class DocumentRecord(BaseModel):
    """Top-level canonical document representation."""
    document_id: str
    metadata: DocumentMetadata
    versions: List[DocumentVersion] = Field(default_factory=list)
    current_version_id: str


# ============================================================
# CONFLICT & RELATIONSHIP SCHEMAS
# ============================================================

class ConflictRecord(BaseModel):
    """Structured record of a detected discrepancy between sources."""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    topic_or_entity: str
    source_a: str
    claim_a: str
    authority_a: SourceAuthority
    date_a: Optional[str] = None
    source_b: str
    claim_b: str
    authority_b: SourceAuthority
    date_b: Optional[str] = None
    evidence: str
    is_resolved: bool = False
    resolution_summary: Optional[str] = None
    superseding_source: Optional[str] = None
    requires_human_clarification: bool = False


class RelationshipRecord(BaseModel):
    """Discovered relationship connecting two documents or entities."""
    relationship_id: str
    source_a: str
    entity_a: str
    source_b: str
    entity_b: str
    relationship_type: RelationshipType
    evidence: str
    confidence: float = 1.0


class KnowledgeGraphNode(BaseModel):
    """Node in the Entity-Document relationship graph."""
    node_id: str
    label: str
    node_type: str  # "entity", "document", "chunk", "version"
    properties: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphEdge(BaseModel):
    """Directed edge in the knowledge graph."""
    edge_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: RelationshipType
    properties: Dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0


# ============================================================
# RETRIEVAL & PROVENANCE SCHEMAS
# ============================================================

class RetrievalFilter(BaseModel):
    """Deterministic security & metadata filter applied prior to retrieval."""
    tenant_id: str = "default_tenant"
    user_access_scopes: List[str] = Field(default_factory=lambda: ["public"])
    document_types: Optional[List[DocumentType]] = None
    departments: Optional[List[str]] = None
    authorities: Optional[List[SourceAuthority]] = None
    effective_year: Optional[int] = None
    only_current_versions: bool = True


class RetrievalCandidate(BaseModel):
    """Retrieved candidate chunk scored across search stages."""
    chunk: DocumentChunk
    bm25_score: float = 0.0
    bm25_rank: Optional[int] = None
    dense_score: float = 0.0
    dense_rank: Optional[int] = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0


class CitationRecord(BaseModel):
    """Explicit citation supporting an answer claim."""
    claim: str
    document_id: str
    version_id: str
    chunk_id: str
    source_filename: str
    authority: SourceAuthority
    supporting_excerpt: str


class ProvenanceChain(BaseModel):
    """Full hierarchical provenance linking an answer back to original source artifacts."""
    answer_summary: str
    citations: List[CitationRecord] = Field(default_factory=list)
    contributing_sources: List[str] = Field(default_factory=list)
    contributing_documents: List[str] = Field(default_factory=list)
    contributing_chunks: List[str] = Field(default_factory=list)


# ============================================================
# STRUCTURED SHARED STATE / SHARED MEMORY
# ============================================================

class TaskContext(BaseModel):
    """User intent, constraints, and success criteria."""
    user_request: str
    interpreted_goal: str
    task_type: str = "rag_query"
    is_complex_query: bool = False
    requires_agentic_reasoning: bool = False
    requires_graph_traversal: bool = False
    constraints: Dict[str, Any] = Field(default_factory=dict)
    success_criteria: List[str] = Field(default_factory=list)


class SourceContext(BaseModel):
    """Registered sources, metadata, and authorities."""
    documents: Dict[str, DocumentRecord] = Field(default_factory=dict)
    source_authorities: Dict[str, SourceAuthority] = Field(default_factory=dict)
    total_chunks: int = 0


class DataContext(BaseModel):
    """Extracted data schemas, statistical findings, and quality metrics."""
    schemas: Dict[str, Any] = Field(default_factory=dict)
    entities: List[str] = Field(default_factory=list)
    relationships: List[RelationshipRecord] = Field(default_factory=list)


class RetrievalContext(BaseModel):
    """Query representations, filters, and intermediate retrieval outputs."""
    raw_query: str = ""
    rewritten_queries: List[str] = Field(default_factory=list)
    filters: RetrievalFilter = Field(default_factory=RetrievalFilter)
    bm25_candidates: List[RetrievalCandidate] = Field(default_factory=list)
    dense_candidates: List[RetrievalCandidate] = Field(default_factory=list)
    fused_candidates: List[RetrievalCandidate] = Field(default_factory=list)
    reranked_results: List[RetrievalCandidate] = Field(default_factory=list)


class EvidenceContext(BaseModel):
    """Synthesized supporting evidence and provenance."""
    supporting_chunks: List[DocumentChunk] = Field(default_factory=list)
    evidence_scores: Dict[str, float] = Field(default_factory=dict)
    provenance: Optional[ProvenanceChain] = None


class DecisionContext(BaseModel):
    """Orchestrator plan, task execution tracking, and replanning traces."""
    current_plan: List[str] = Field(default_factory=list)
    completed_tasks: List[str] = Field(default_factory=list)
    pending_tasks: List[str] = Field(default_factory=list)
    running_tasks: List[str] = Field(default_factory=list)
    failed_tasks: List[str] = Field(default_factory=list)
    skipped_tasks: List[str] = Field(default_factory=list)
    replanning_reason: Optional[str] = None


class ConflictContext(BaseModel):
    """Identified discrepancies and their resolution status."""
    detected_conflicts: List[ConflictRecord] = Field(default_factory=list)
    resolved_conflicts: List[ConflictRecord] = Field(default_factory=list)
    unresolved_conflicts: List[ConflictRecord] = Field(default_factory=list)


class QualityContext(BaseModel):
    """Confidence scores, validation status, and unresolved gaps."""
    confidence_score: float = 1.0
    validation_status: str = "PASSED"
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    unresolved_gaps: List[str] = Field(default_factory=list)


class SecurityContext(BaseModel):
    """Tenant isolation, access control, and adversarial prompt detection."""
    tenant_id: str = "default_tenant"
    user_id: str = "default_user"
    user_permissions: List[str] = Field(default_factory=lambda: ["read:public"])
    access_scope: List[str] = Field(default_factory=lambda: ["public"])
    security_flags: List[str] = Field(default_factory=list)
    injection_detected: bool = False


class ArtifactContext(BaseModel):
    """Persistent artifacts, files, and outputs."""
    raw_files: List[str] = Field(default_factory=list)
    parsed_files: List[str] = Field(default_factory=list)
    reports: List[str] = Field(default_factory=list)


class RAGSharedState(BaseModel):
    """Complete structured shared state for the Knowledge & RAG platform."""
    task: TaskContext = Field(default_factory=lambda: TaskContext(user_request="", interpreted_goal=""))
    sources: SourceContext = Field(default_factory=SourceContext)
    data: DataContext = Field(default_factory=DataContext)
    retrieval: RetrievalContext = Field(default_factory=RetrievalContext)
    evidence: EvidenceContext = Field(default_factory=EvidenceContext)
    decision: DecisionContext = Field(default_factory=DecisionContext)
    conflicts: ConflictContext = Field(default_factory=ConflictContext)
    quality: QualityContext = Field(default_factory=QualityContext)
    security: SecurityContext = Field(default_factory=SecurityContext)
    artifacts: ArtifactContext = Field(default_factory=ArtifactContext)
