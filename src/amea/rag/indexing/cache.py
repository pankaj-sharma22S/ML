"""Multi-tier cache for ingestion, parsing, embeddings, and security-scoped retrieval."""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple
from amea.rag.models import RetrievalCandidate


class MultiTierRAGCache:
    """Manages Ingestion, Embedding, and Retrieval caching layers."""

    def __init__(self):
        # Ingestion cache: file_hash -> (raw_text, structured_sections, metadata)
        self.ingestion_cache: Dict[str, Tuple[str, List[Dict[str, Any]], Dict[str, Any]]] = {}

        # Embedding cache: (content_hash, model, version) -> List[float]
        self.embedding_cache: Dict[str, List[float]] = {}

        # Retrieval cache: cache_key -> List[RetrievalCandidate]
        self.retrieval_cache: Dict[str, List[RetrievalCandidate]] = {}

        self.stats = {
            "ingestion_hits": 0,
            "ingestion_misses": 0,
            "embedding_hits": 0,
            "embedding_misses": 0,
            "retrieval_hits": 0,
            "retrieval_misses": 0,
        }

    # --- Ingestion Cache ---
    def get_parsed_document(self, file_hash: str) -> Optional[Tuple[str, List[Dict[str, Any]], Dict[str, Any]]]:
        if file_hash in self.ingestion_cache:
            self.stats["ingestion_hits"] += 1
            return self.ingestion_cache[file_hash]
        self.stats["ingestion_misses"] += 1
        return None

    def put_parsed_document(self, file_hash: str, data: Tuple[str, List[Dict[str, Any]], Dict[str, Any]]):
        self.ingestion_cache[file_hash] = data

    # --- Embedding Cache ---
    def get_embedding(self, content_hash: str, model: str = "default", version: str = "v1") -> Optional[List[float]]:
        key = f"{content_hash}_{model}_{version}"
        if key in self.embedding_cache:
            self.stats["embedding_hits"] += 1
            return self.embedding_cache[key]
        self.stats["embedding_misses"] += 1
        return None

    def put_embedding(self, content_hash: str, embedding: List[float], model: str = "default", version: str = "v1"):
        key = f"{content_hash}_{model}_{version}"
        self.embedding_cache[key] = embedding

    # --- Retrieval Cache ---
    def _make_retrieval_key(self, query: str, tenant_id: str, scopes: List[str], index_version: str) -> str:
        payload = {
            "q": query.strip().lower(),
            "t": tenant_id,
            "s": sorted(scopes),
            "v": index_version,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def get_retrieval(self, query: str, tenant_id: str, scopes: List[str], index_version: str = "v1") -> Optional[List[RetrievalCandidate]]:
        key = self._make_retrieval_key(query, tenant_id, scopes, index_version)
        if key in self.retrieval_cache:
            self.stats["retrieval_hits"] += 1
            return self.retrieval_cache[key]
        self.stats["retrieval_misses"] += 1
        return None

    def put_retrieval(self, query: str, tenant_id: str, scopes: List[str], candidates: List[RetrievalCandidate], index_version: str = "v1"):
        key = self._make_retrieval_key(query, tenant_id, scopes, index_version)
        self.retrieval_cache[key] = candidates

    def invalidate_retrieval(self):
        """Invalidate retrieval cache when corpus or index updates."""
        self.retrieval_cache.clear()

    @property
    def hit_rate(self) -> float:
        total_hits = sum([self.stats["ingestion_hits"], self.stats["embedding_hits"], self.stats["retrieval_hits"]])
        total_misses = sum([self.stats["ingestion_misses"], self.stats["embedding_misses"], self.stats["retrieval_misses"]])
        total = total_hits + total_misses
        return total_hits / total if total > 0 else 0.0
