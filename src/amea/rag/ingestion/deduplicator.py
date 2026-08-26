"""Deduplication and near-duplicate detection engine using SHA-256 and MinHash/Jaccard."""

import hashlib
import re
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum


class DuplicateStatus(str, Enum):
    EXACT_DUPLICATE = "EXACT_DUPLICATE"
    NEAR_DUPLICATE = "NEAR_DUPLICATE"
    VERSION_CANDIDATE = "VERSION_CANDIDATE"
    RELATED_DOCUMENT = "RELATED_DOCUMENT"
    UNIQUE = "UNIQUE"


class DocumentDeduplicator:
    """Detects exact byte duplicates and near-duplicate text corpora."""

    @staticmethod
    def compute_sha256(content: str) -> str:
        """Compute exact SHA-256 hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text by lowering and collapsing whitespace/punctuation."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def compute_normalized_hash(cls, text: str) -> str:
        """Compute SHA-256 hash of normalized text."""
        norm = cls.normalize_text(text)
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    @classmethod
    def get_shingles(cls, text: str, k: int = 3) -> Set[str]:
        """Extract word k-shingles for Jaccard / MinHash comparison."""
        words = cls.normalize_text(text).split()
        if len(words) < k:
            return {" ".join(words)}
        return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}

    @classmethod
    def compute_jaccard_similarity(cls, text_a: str, text_b: str, k: int = 3) -> float:
        """Compute Jaccard similarity between two texts based on word shingles."""
        set_a = cls.get_shingles(text_a, k)
        set_b = cls.get_shingles(text_b, k)
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    @classmethod
    def evaluate_duplicate_status(
        cls,
        new_content: str,
        existing_corpus: Dict[str, str], # doc_id -> content
        near_duplicate_threshold: float = 0.85,
        related_threshold: float = 0.50,
    ) -> Tuple[DuplicateStatus, Optional[str], float]:
        """
        Check new document against existing corpus.
        Returns (DuplicateStatus, matched_doc_id, similarity_score).
        """
        new_norm_hash = cls.compute_normalized_hash(new_content)

        # 1. Exact hash check
        for doc_id, content in existing_corpus.items():
            if cls.compute_normalized_hash(content) == new_norm_hash:
                return DuplicateStatus.EXACT_DUPLICATE, doc_id, 1.0

        # 2. Near-duplicate Jaccard similarity check
        best_match_id = None
        best_sim = 0.0

        for doc_id, content in existing_corpus.items():
            sim = cls.compute_jaccard_similarity(new_content, content)
            if sim > best_sim:
                best_sim = sim
                best_match_id = doc_id

        if best_sim >= near_duplicate_threshold:
            return DuplicateStatus.NEAR_DUPLICATE, best_match_id, best_sim
        elif best_sim >= related_threshold:
            return DuplicateStatus.RELATED_DOCUMENT, best_match_id, best_sim

        return DuplicateStatus.UNIQUE, None, 0.0
