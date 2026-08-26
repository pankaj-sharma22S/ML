"""Hybrid search orchestrator fusing BM25 lexical and Dense ANN vector rankings via RRF."""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from amea.rag.indexing.ann_index import DenseANNIndex
from amea.rag.indexing.bm25_index import BM25Index
from amea.rag.models import DocumentChunk, RetrievalCandidate, RetrievalFilter
from amea.rag.retrieval.filter import PreRetrievalFilter


class HybridRetriever:
    """Combines BM25 and Dense ANN vector searches using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        bm25_index: BM25Index,
        ann_index: DenseANNIndex,
        all_chunks: Dict[str, DocumentChunk],
        rrf_k: int = 60,
    ):
        self.bm25_index = bm25_index
        self.ann_index = ann_index
        self.all_chunks = all_chunks
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        filters: Optional[RetrievalFilter] = None,
        top_candidates_count: int = 50,
    ) -> List[RetrievalCandidate]:
        """Execute pre-filtered parallel hybrid search and RRF fusion."""
        active_filters = filters or RetrievalFilter()
        allowed_ids = PreRetrievalFilter.get_allowed_chunk_ids(self.all_chunks, active_filters)

        if not allowed_ids:
            return []

        # Parallel search execution
        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self.bm25_index.search, query, top_candidates_count, allowed_ids)
            dense_future = executor.submit(self.ann_index.search, query, top_candidates_count, allowed_ids)

            bm25_results = bm25_future.result()
            dense_results = dense_future.result()

        # Reciprocal Rank Fusion (RRF) map: chunk_id -> Candidate
        candidates_map: Dict[str, RetrievalCandidate] = {}

        # Process BM25 rankings
        for cand in bm25_results:
            cid = cand.chunk.chunk_id
            rrf_val = 1.0 / (self.rrf_k + (cand.bm25_rank or 999))
            candidates_map[cid] = RetrievalCandidate(
                chunk=cand.chunk,
                bm25_score=cand.bm25_score,
                bm25_rank=cand.bm25_rank,
                rrf_score=rrf_val,
            )

        # Process Dense rankings
        for cand in dense_results:
            cid = cand.chunk.chunk_id
            rrf_val = 1.0 / (self.rrf_k + (cand.dense_rank or 999))
            if cid in candidates_map:
                existing = candidates_map[cid]
                existing.dense_score = cand.dense_score
                existing.dense_rank = cand.dense_rank
                existing.rrf_score += rrf_val
            else:
                candidates_map[cid] = RetrievalCandidate(
                    chunk=cand.chunk,
                    dense_score=cand.dense_score,
                    dense_rank=cand.dense_rank,
                    rrf_score=rrf_val,
                )

        # Sort by fused RRF score descending
        sorted_fused = sorted(candidates_map.values(), key=lambda c: c.rrf_score, reverse=True)
        return sorted_fused[:top_candidates_count]
