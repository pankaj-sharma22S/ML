"""Lexical BM25 search index for exact term, error code, and keyword retrieval."""

import math
import re
from typing import Dict, List, Optional, Set, Tuple
from amea.rag.models import DocumentChunk, RetrievalCandidate


class BM25Index:
    """Okapi BM25 Lexical Inverted Index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: Dict[str, DocumentChunk] = {} # chunk_id -> DocumentChunk
        self.doc_lengths: Dict[str, int] = {}     # chunk_id -> length
        self.avg_doc_len: float = 0.0
        self.inverted_index: Dict[str, Dict[str, int]] = {} # term -> {chunk_id: tf}
        self.doc_frequencies: Dict[str, int] = {} # term -> df
        self.total_documents: int = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize and normalize text preserving alphanumeric identifiers."""
        return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Batch index chunks into BM25 inverted index."""
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            tokens = self.tokenize(chunk.content)
            doc_len = len(tokens)
            self.doc_lengths[chunk.chunk_id] = doc_len

            term_counts: Dict[str, int] = {}
            for t in tokens:
                term_counts[t] = term_counts.get(t, 0) + 1

            for term, count in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                    self.doc_frequencies[term] = 0
                self.inverted_index[term][chunk.chunk_id] = count
                self.doc_frequencies[term] += 1

        self.total_documents = len(self.chunks)
        if self.total_documents > 0:
            self.avg_doc_len = sum(self.doc_lengths.values()) / self.total_documents

    def search(
        self,
        query: str,
        top_n: int = 20,
        allowed_chunk_ids: Optional[Set[str]] = None,
    ) -> List[RetrievalCandidate]:
        """Search query against BM25 index with optional pre-filtered chunk IDs."""
        query_tokens = self.tokenize(query)
        if not query_tokens or self.total_documents == 0:
            return []

        scores: Dict[str, float] = {}

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            df = self.doc_frequencies[term]
            # IDF with smoothing
            idf = math.log(1.0 + (self.total_documents - df + 0.5) / (df + 0.5))

            for chunk_id, tf in self.inverted_index[term].items():
                if allowed_chunk_ids is not None and chunk_id not in allowed_chunk_ids:
                    continue

                doc_len = self.doc_lengths.get(chunk_id, 1)
                len_norm = 1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0))
                term_score = idf * ((tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm))
                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

        # Sort by score descending
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        for rank, (chunk_id, score) in enumerate(sorted_items, start=1):
            chunk = self.chunks[chunk_id]
            results.append(RetrievalCandidate(
                chunk=chunk,
                bm25_score=score,
                bm25_rank=rank,
            ))

        return results
