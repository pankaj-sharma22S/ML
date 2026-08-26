"""Adaptive RAG Agent coordinating hybrid search, graph traversal, conflict resolution, and provenance."""

from typing import Dict, List, Optional
from amea.rag.agentic.adaptive_planner import AdaptiveQueryPlanner
from amea.rag.conflicts.detector import MultiDocumentConflictDetector
from amea.rag.conflicts.resolver import ConflictResolver
from amea.rag.graph.knowledge_graph import KnowledgeGraph
from amea.rag.graph.relationship_miner import MultiDocumentRelationshipMiner
from amea.rag.indexing.ann_index import DenseANNIndex
from amea.rag.indexing.bm25_index import BM25Index
from amea.rag.models import (
    CitationRecord,
    ConflictRecord,
    DocumentChunk,
    DocumentRecord,
    ProvenanceChain,
    RetrievalCandidate,
    RetrievalFilter,
    SourceAuthority,
)
from amea.rag.retrieval.hybrid_retriever import HybridRetriever
from amea.rag.retrieval.reranker import CrossEncoderReranker


class AdaptiveRAGAgent:
    """Enterprise RAG Agent executing dynamic routing, hybrid retrieval, conflict resolution, and provenance."""

    def __init__(
        self,
        bm25_index: BM25Index,
        ann_index: DenseANNIndex,
        all_chunks: Dict[str, DocumentChunk],
        documents: Dict[str, DocumentRecord],
        knowledge_graph: Optional[KnowledgeGraph] = None,
    ):
        self.bm25_index = bm25_index
        self.ann_index = ann_index
        self.all_chunks = all_chunks
        self.documents = documents
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.retriever = HybridRetriever(bm25_index, ann_index, all_chunks)
        self.reranker = CrossEncoderReranker(top_k=5)

    def query(
        self,
        user_query: str,
        filters: Optional[RetrievalFilter] = None,
    ) -> Dict:
        """Executes adaptive search path and returns structured evidence with full provenance."""
        active_filters = filters or RetrievalFilter()
        is_complex, complexity_reasons = AdaptiveQueryPlanner.analyze_query(user_query)

        # 1. Hybrid Retrieval & Reranking
        raw_candidates = self.retriever.retrieve(
            query=user_query,
            filters=active_filters,
            top_candidates_count=30 if is_complex else 15,
        )

        authorities_map = {
            doc_id: doc.metadata.authority for doc_id, doc in self.documents.items()
        }

        top_candidates = self.reranker.rerank(
            query=user_query,
            candidates=raw_candidates,
            authorities_map=authorities_map,
        )

        selected_chunks = [c.chunk for c in top_candidates]

        # 2. If complex / multi-document, perform conflict check & graph reasoning
        detected_conflicts: List[ConflictRecord] = []
        resolved_conflicts: List[ConflictRecord] = []
        unresolved_conflicts: List[ConflictRecord] = []
        graph_entities: List[str] = []

        if is_complex and len(selected_chunks) > 1:
            # Conflict analysis
            detected_conflicts = MultiDocumentConflictDetector.detect_conflicts(
                chunks=selected_chunks,
                authorities=authorities_map,
            )
            if detected_conflicts:
                resolved_conflicts, unresolved_conflicts = ConflictResolver.resolve_conflicts(detected_conflicts)

            # Knowledge Graph traversal for query terms
            for term in user_query.split():
                if term in self.knowledge_graph.nodes:
                    connected = self.knowledge_graph.multi_hop_traverse(term, max_hops=2)
                    graph_entities.extend([n.label for n in connected])

        # 3. Construct Provenance Chain
        citations: List[CitationRecord] = []
        contributing_sources = set()
        contributing_docs = set()
        contributing_chunks = []

        for cand in top_candidates:
            chk = cand.chunk
            doc_id = chk.metadata.document_id
            doc = self.documents.get(doc_id)
            fn = doc.metadata.filename if doc else chk.metadata.source_uri
            auth = authorities_map.get(doc_id, SourceAuthority.INTERNAL_DOCUMENTATION)

            citations.append(CitationRecord(
                claim=chk.content[:120] + "...",
                document_id=doc_id,
                version_id=chk.metadata.version_id,
                chunk_id=chk.chunk_id,
                source_filename=fn,
                authority=auth,
                supporting_excerpt=chk.content,
            ))
            contributing_sources.add(fn)
            contributing_docs.add(doc_id)
            contributing_chunks.append(chk.chunk_id)

        provenance = ProvenanceChain(
            answer_summary=f"Synthesized evidence from {len(top_candidates)} top-ranked chunks across {len(contributing_docs)} documents.",
            citations=citations,
            contributing_sources=list(contributing_sources),
            contributing_documents=list(contributing_docs),
            contributing_chunks=contributing_chunks,
        )

        return {
            "query": user_query,
            "is_complex_query": is_complex,
            "complexity_reasons": complexity_reasons,
            "top_candidates_count": len(top_candidates),
            "top_candidates": top_candidates,
            "conflicts_detected_count": len(detected_conflicts),
            "resolved_conflicts": resolved_conflicts,
            "unresolved_conflicts": unresolved_conflicts,
            "graph_entities_discovered": list(set(graph_entities)),
            "provenance": provenance,
        }
