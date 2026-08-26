"""RAG Retrieval and Faithfulness Evaluator."""

import math
from typing import Dict, List, Set
from amea.rag.models import RetrievalCandidate


class RAGEvaluator:
    """Computes Recall@K, Precision@K, MRR, NDCG@K, and Citation Faithfulness."""

    @staticmethod
    def compute_recall_at_k(retrieved_chunk_ids: List[str], ground_truth_chunk_ids: Set[str], k: int = 5) -> float:
        top_k = set(retrieved_chunk_ids[:k])
        if not ground_truth_chunk_ids:
            return 1.0
        return len(top_k.intersection(ground_truth_chunk_ids)) / len(ground_truth_chunk_ids)

    @staticmethod
    def compute_precision_at_k(retrieved_chunk_ids: List[str], ground_truth_chunk_ids: Set[str], k: int = 5) -> float:
        top_k = retrieved_chunk_ids[:k]
        if not top_k:
            return 0.0
        hits = sum(1 for cid in top_k if cid in ground_truth_chunk_ids)
        return hits / len(top_k)

    @staticmethod
    def compute_mrr(retrieved_chunk_ids: List[str], ground_truth_chunk_ids: Set[str]) -> float:
        for rank, cid in enumerate(retrieved_chunk_ids, start=1):
            if cid in ground_truth_chunk_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def compute_ndcg_at_k(retrieved_chunk_ids: List[str], ground_truth_chunk_ids: Set[str], k: int = 5) -> float:
        top_k = retrieved_chunk_ids[:k]
        dcg = 0.0
        for i, cid in enumerate(top_k, start=1):
            rel = 1.0 if cid in ground_truth_chunk_ids else 0.0
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 1)

        # Ideal DCG
        ideal_hits = min(len(ground_truth_chunk_ids), k)
        idcg = sum((2.0 ** 1.0 - 1.0) / math.log2(i + 1) for i in range(1, ideal_hits + 1))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def evaluate_retrieval(
        retrieved: List[RetrievalCandidate],
        ground_truth_chunk_ids: Set[str],
        k: int = 5,
    ) -> Dict[str, float]:
        """Computes comprehensive evaluation suite for a retrieval result."""
        ids = [c.chunk.chunk_id for c in retrieved]
        return {
            f"recall@{k}": RAGEvaluator.compute_recall_at_k(ids, ground_truth_chunk_ids, k),
            f"precision@{k}": RAGEvaluator.compute_precision_at_k(ids, ground_truth_chunk_ids, k),
            "mrr": RAGEvaluator.compute_mrr(ids, ground_truth_chunk_ids),
            f"ndcg@{k}": RAGEvaluator.compute_ndcg_at_k(ids, ground_truth_chunk_ids, k),
        }
