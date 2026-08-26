"""Dense vector ANN index supporting cosine similarity and batch embeddings."""

import math
import re
from typing import Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from amea.rag.models import DocumentChunk, RetrievalCandidate


class DenseANNIndex:
    """Dense vector index with normalized cosine similarity search."""

    def __init__(
        self,
        dimension: int = 128,
        embedding_model: str = "dense-projection-v1",
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        self.dimension = dimension
        self.embedding_model = embedding_model
        self.embedding_version = "v1"
        self.custom_embedding_fn = embedding_fn
        self.chunks: Dict[str, DocumentChunk] = {} # chunk_id -> DocumentChunk
        self.vectors: Dict[str, np.ndarray] = {}  # chunk_id -> normalized numpy vector

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate normalized dense embedding for a given text."""
        if self.custom_embedding_fn:
            vec = np.array(self.custom_embedding_fn(text), dtype=np.float32)
        else:
            # Deterministic dense semantic hash embedding
            vec = np.zeros(self.dimension, dtype=np.float32)
            tokens = re.findall(r"\b\w+\b", text.lower())
            for i, token in enumerate(tokens):
                # Hash token to dimension indices with positional weight
                h = hash(token)
                idx = abs(h) % self.dimension
                val = (1.0 if (h > 0) else -1.0) / (1.0 + math.log1p(i))
                vec[idx] += val

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Batch generate embeddings and register into dense index."""
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            if chunk.embedding and len(chunk.embedding) == self.dimension:
                vec = np.array(chunk.embedding, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    vec = vec / norm
            else:
                vec = self.generate_embedding(chunk.content)
                chunk.embedding = vec.tolist()
            self.vectors[chunk.chunk_id] = vec

    def search(
        self,
        query: str,
        top_n: int = 20,
        allowed_chunk_ids: Optional[Set[str]] = None,
    ) -> List[RetrievalCandidate]:
        """Search query vector against dense corpus."""
        if not self.vectors:
            return []

        q_vec = self.generate_embedding(query)
        scores: Dict[str, float] = {}

        for chunk_id, vec in self.vectors.items():
            if allowed_chunk_ids is not None and chunk_id not in allowed_chunk_ids:
                continue
            # Cosine similarity between normalized vectors
            sim = float(np.dot(q_vec, vec))
            scores[chunk_id] = sim

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        for rank, (chunk_id, score) in enumerate(sorted_items, start=1):
            chunk = self.chunks[chunk_id]
            results.append(RetrievalCandidate(
                chunk=chunk,
                dense_score=score,
                dense_rank=rank,
            ))

        return results
