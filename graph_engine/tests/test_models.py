"""
Tests for data contracts defined in graph-design-spec.md §2 (Node / Edge / Triple).

Rules:
- Tests derive from spec only — no implementation code was read.
- Each test checks one observable behaviour through the public interface.
"""

import pytest
from graph_engine.models import Node, Edge, Triple


# ---------------------------------------------------------------------------
# Node — spec §2.1
# ---------------------------------------------------------------------------

class TestNodeSchema:
    """Node must satisfy the field contract in spec §2.1."""

    def _make_node(self, name="SensorA", node_type="Device", source_doc="doc1.txt"):
        return Node(name=name, type=node_type, source_doc=source_doc)

    def test_node_has_id(self):
        node = self._make_node()
        assert hasattr(node, "id") and node.id, "Node must have a non-empty id"

    def test_node_has_name(self):
        node = self._make_node(name="SensorA")
        assert node.name == "SensorA"

    def test_node_has_type(self):
        node = self._make_node(node_type="Device")
        assert node.type == "Device"

    def test_node_has_source_doc(self):
        node = self._make_node(source_doc="manual.pdf")
        assert node.source_doc == "manual.pdf"

    # spec §2.1 — id is "UUID or hash of name + type"
    def test_node_id_is_deterministic_same_name_and_type(self):
        """Same name + type must always produce the same id (spec §2.1)."""
        n1 = Node(name="SensorA", type="Device", source_doc="doc1.txt")
        n2 = Node(name="SensorA", type="Device", source_doc="doc2.txt")
        assert n1.id == n2.id, "id must depend only on name + type, not source_doc"

    def test_node_id_differs_when_type_differs(self):
        """Same name but different type must produce a different id (spec §2.1)."""
        device_node = Node(name="SensorA", type="Device", source_doc="doc1.txt")
        component_node = Node(name="SensorA", type="Component", source_doc="doc1.txt")
        assert device_node.id != component_node.id

    def test_node_id_differs_when_name_differs(self):
        n1 = Node(name="SensorA", type="Device", source_doc="doc1.txt")
        n2 = Node(name="SensorB", type="Device", source_doc="doc1.txt")
        assert n1.id != n2.id


# ---------------------------------------------------------------------------
# Edge — spec §2.2
# ---------------------------------------------------------------------------

class TestEdgeSchema:
    """Edge must satisfy the field contract in spec §2.2."""

    def _make_edge(self, relation="part_of", weight=0.9):
        subject = Node(name="SensorA", type="Device", source_doc="doc1.txt")
        obj = Node(name="ControlUnit", type="Component", source_doc="doc1.txt")
        return Edge(
            source=subject.id,
            target=obj.id,
            relation=relation,
            weight=weight,
            source_doc="doc1.txt",
        )

    def test_edge_has_source(self):
        edge = self._make_edge()
        assert hasattr(edge, "source") and edge.source

    def test_edge_has_target(self):
        edge = self._make_edge()
        assert hasattr(edge, "target") and edge.target

    def test_edge_has_relation(self):
        edge = self._make_edge(relation="part_of")
        assert edge.relation == "part_of"

    def test_edge_has_weight(self):
        edge = self._make_edge(weight=0.8)
        assert edge.weight == pytest.approx(0.8)

    def test_edge_has_source_doc(self):
        subject = Node(name="SensorA", type="Device", source_doc="doc1.txt")
        obj = Node(name="ControlUnit", type="Component", source_doc="doc1.txt")
        edge = Edge(
            source=subject.id, target=obj.id,
            relation="part_of", weight=0.9, source_doc="manual.pdf"
        )
        assert edge.source_doc == "manual.pdf"

    def test_edge_source_matches_subject_node_id(self):
        """Edge.source must equal the subject Node's id (spec §2.2)."""
        subject = Node(name="SensorA", type="Device", source_doc="doc1.txt")
        obj = Node(name="ControlUnit", type="Component", source_doc="doc1.txt")
        edge = Edge(
            source=subject.id, target=obj.id,
            relation="part_of", weight=0.9, source_doc="doc1.txt"
        )
        assert edge.source == subject.id

    def test_edge_target_matches_object_node_id(self):
        """Edge.target must equal the object Node's id (spec §2.2)."""
        subject = Node(name="SensorA", type="Device", source_doc="doc1.txt")
        obj = Node(name="ControlUnit", type="Component", source_doc="doc1.txt")
        edge = Edge(
            source=subject.id, target=obj.id,
            relation="part_of", weight=0.9, source_doc="doc1.txt"
        )
        assert edge.target == obj.id

    def test_edge_weight_low_boundary(self):
        """Weight at 0.0 is accepted (spec §2.2 — range 0.0–1.0)."""
        edge = self._make_edge(weight=0.0)
        assert edge.weight == pytest.approx(0.0)

    def test_edge_weight_high_boundary(self):
        """Weight at 1.0 is accepted (spec §2.2)."""
        edge = self._make_edge(weight=1.0)
        assert edge.weight == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Triple — spec §2 + §5
# ---------------------------------------------------------------------------

class TestTripleSchema:
    """Triple carries subject Node, predicate str, object Node, weight, source_doc."""

    def _make_triple(self, weight=0.85, source_doc="doc1.txt"):
        subject = Node(name="SensorA", type="Device", source_doc=source_doc)
        obj = Node(name="ControlUnit", type="Component", source_doc=source_doc)
        return Triple(
            subject=subject,
            predicate="part_of",
            object=obj,
            weight=weight,
            source_doc=source_doc,
        )

    def test_triple_has_subject_node(self):
        triple = self._make_triple()
        assert isinstance(triple.subject, Node)

    def test_triple_has_object_node(self):
        triple = self._make_triple()
        assert isinstance(triple.object, Node)

    def test_triple_has_predicate(self):
        triple = self._make_triple()
        assert triple.predicate == "part_of"

    def test_triple_carries_weight(self):
        """spec §5 — weight must always be included on triple output."""
        triple = self._make_triple(weight=0.75)
        assert triple.weight == pytest.approx(0.75)

    def test_triple_carries_source_doc(self):
        """spec §5 — source_doc must always be included on triple output."""
        triple = self._make_triple(source_doc="manual_v2.pdf")
        assert triple.source_doc == "manual_v2.pdf"
