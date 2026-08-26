"""Knowledge Graph representation supporting multi-hop entity traversal."""

from collections import deque
from typing import Dict, List, Optional, Set
from uuid import uuid4
from amea.rag.models import (
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    RelationshipRecord,
    RelationshipType,
)


class KnowledgeGraph:
    """In-memory Knowledge Graph for entity-document traversal."""

    def __init__(self):
        self.nodes: Dict[str, KnowledgeGraphNode] = {} # node_id -> Node
        self.edges: Dict[str, List[KnowledgeGraphEdge]] = {} # source_node_id -> list of Edges

    def add_node(self, node_id: str, label: str, node_type: str, properties: Optional[Dict] = None):
        if node_id not in self.nodes:
            self.nodes[node_id] = KnowledgeGraphNode(
                node_id=node_id,
                label=label,
                node_type=node_type,
                properties=properties or {},
            )
            self.edges[node_id] = []

    def add_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        relationship_type: RelationshipType,
        properties: Optional[Dict] = None,
    ):
        self.add_node(source_node_id, source_node_id, "unknown")
        self.add_node(target_node_id, target_node_id, "unknown")

        edge = KnowledgeGraphEdge(
            edge_id=f"edge_{uuid4().hex[:6]}",
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_type=relationship_type,
            properties=properties or {},
        )
        self.edges[source_node_id].append(edge)

    def populate_from_relationships(self, relationships: List[RelationshipRecord]):
        """Populate graph from mined relationship records."""
        for rel in relationships:
            # Source Doc -> Entity
            self.add_node(rel.source_a, rel.source_a, "document")
            self.add_node(rel.entity_a, rel.entity_a, "entity")
            self.add_edge(rel.source_a, rel.entity_a, rel.relationship_type)

            # Entity -> Target Doc
            self.add_node(rel.source_b, rel.source_b, "document")
            self.add_node(rel.entity_b, rel.entity_b, "entity")
            self.add_edge(rel.entity_a, rel.source_b, rel.relationship_type)

    def multi_hop_traverse(self, start_node_id: str, max_hops: int = 2) -> List[KnowledgeGraphNode]:
        """Perform multi-hop traversal to retrieve connected entities and documents."""
        if start_node_id not in self.nodes:
            return []

        visited: Set[str] = {start_node_id}
        queue = deque([(start_node_id, 0)])
        result_nodes: List[KnowledgeGraphNode] = []

        while queue:
            curr_id, depth = queue.popleft()
            if depth > 0:
                result_nodes.append(self.nodes[curr_id])

            if depth < max_hops:
                for edge in self.edges.get(curr_id, []):
                    target = edge.target_node_id
                    if target not in visited:
                        visited.add(target)
                        queue.append((target, depth + 1))

        return result_nodes
