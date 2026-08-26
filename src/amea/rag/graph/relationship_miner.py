"""Relationship Miner discovering structural, semantic, temporal, and entity relationships."""

import re
from typing import Dict, List, Set
from uuid import uuid4
from amea.rag.models import (
    DocumentRecord,
    RelationshipRecord,
    RelationshipType,
)


class MultiDocumentRelationshipMiner:
    """Discovers relationships across independent documents without assuming pre-existing connections."""

    @staticmethod
    def extract_entities(text: str) -> Set[str]:
        """Extract capitalized keywords, product names, error codes, and technical entities."""
        # Alphanumeric identifiers, capitalized terms, or capitalized pairs
        matches = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]{2,}\b", text)
        stop_words = {"The", "This", "That", "When", "What", "Where", "With", "From", "Have", "Will", "Each", "All"}
        return {m for m in matches if m not in stop_words}

    @classmethod
    def mine_relationships(
        cls,
        documents: Dict[str, DocumentRecord],
        document_texts: Dict[str, str], # doc_id -> full text
    ) -> List[RelationshipRecord]:
        """Mine multi-document relationships across the corpus."""
        relationships: List[RelationshipRecord] = []
        doc_ids = list(documents.keys())

        # Extract entities per document
        doc_entities: Dict[str, Set[str]] = {}
        for doc_id, text in document_texts.items():
            doc_entities[doc_id] = cls.extract_entities(text)

        # Pairwise comparison
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                id_a, id_b = doc_ids[i], doc_ids[j]
                doc_a, doc_b = documents[id_a], documents[id_b]
                ent_a, ent_b = doc_entities.get(id_a, set()), doc_entities.get(id_b, set())

                # 1. Semantic shared entities
                shared = ent_a.intersection(ent_b)
                for ent in shared:
                    relationships.append(RelationshipRecord(
                        relationship_id=f"rel_{uuid4().hex[:6]}",
                        source_a=id_a,
                        entity_a=ent,
                        source_b=id_b,
                        entity_b=ent,
                        relationship_type=RelationshipType.SEMANTIC,
                        evidence=f"Both '{doc_a.metadata.filename}' and '{doc_b.metadata.filename}' reference entity '{ent}'.",
                        confidence=0.85,
                    ))

                # 2. Document citation / references check
                text_a = document_texts.get(id_a, "").lower()
                text_b = document_texts.get(id_b, "").lower()

                if doc_b.metadata.filename.lower() in text_a:
                    relationships.append(RelationshipRecord(
                        relationship_id=f"rel_{uuid4().hex[:6]}",
                        source_a=id_a,
                        entity_a=doc_a.metadata.filename,
                        source_b=id_b,
                        entity_b=doc_b.metadata.filename,
                        relationship_type=RelationshipType.DOCUMENT_REF,
                        evidence=f"'{doc_a.metadata.filename}' explicitly cites/references '{doc_b.metadata.filename}'.",
                        confidence=0.95,
                    ))

                # 3. Temporal version relationship check
                if (doc_a.metadata.filename.split(".")[0] in doc_b.metadata.filename) or \
                   ("2024" in doc_a.metadata.filename and "2025" in doc_b.metadata.filename):
                    relationships.append(RelationshipRecord(
                        relationship_id=f"rel_{uuid4().hex[:6]}",
                        source_a=id_a,
                        entity_a="TimePeriod",
                        source_b=id_b,
                        entity_b="TimePeriod",
                        relationship_type=RelationshipType.TEMPORAL,
                        evidence=f"Temporal sequence detected between '{doc_a.metadata.filename}' and '{doc_b.metadata.filename}'.",
                        confidence=0.90,
                    ))

        return relationships
