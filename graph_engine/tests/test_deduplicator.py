"""
Tests for deduplication logic.

Rules derive from:
  - graph-design-spec.md §7 (Pipeline — Node deduplication):
      "Automatic — exact match on name + type → merge into one node"
  - graph-design-spec.md §8 Decisions log (2026-05-20):
      "Node deduplication: automatic — exact match on name + type → merge"
  - graph-generation-analysis.md (deduplicator spec):
      - Union source_docs (comma-separated)
      - Keep highest weight when same (subject, predicate, object) triple appears twice

No implementation code was read.
"""

import pytest
from graph_engine.models import Node, Triple
from graph_engine.deduplicator import deduplicate_triples


def _make_triple(
    subj_name, subj_type, pred, obj_name, obj_type,
    weight=0.8, source_doc="doc1.txt"
):
    subject = Node(name=subj_name, type=subj_type, source_doc=source_doc)
    obj = Node(name=obj_name, type=obj_type, source_doc=source_doc)
    return Triple(
        subject=subject,
        predicate=pred,
        object=obj,
        weight=weight,
        source_doc=source_doc,
    )


class TestDeduplicateNodes:
    """Node merge rules — spec §7."""

    def test_empty_input_returns_empty_list(self):
        assert deduplicate_triples([]) == []

    def test_single_triple_is_returned_unchanged(self):
        triples = [_make_triple("SensorA", "Device", "part_of", "Unit", "Component")]
        result = deduplicate_triples(triples)
        assert len(result) == 1

    def test_two_triples_with_different_nodes_both_kept(self):
        t1 = _make_triple("SensorA", "Device", "part_of", "UnitX", "Component")
        t2 = _make_triple("SensorB", "Device", "part_of", "UnitY", "Component")
        result = deduplicate_triples([t1, t2])
        assert len(result) == 2

    def test_same_name_and_type_merges_into_one_subject_node(self):
        """
        Two triples that reference the same subject (same name+type) from
        different documents must share the canonical subject node id.
        spec §7: exact match on name + type → merge into one node.
        """
        t1 = _make_triple("SensorA", "Device", "part_of", "Unit", "Component", source_doc="doc1.txt")
        t2 = _make_triple("SensorA", "Device", "causes", "Error42", "Error", source_doc="doc2.txt")
        result = deduplicate_triples([t1, t2])

        subject_ids = {t.subject.id for t in result}
        assert len(subject_ids) == 1, (
            "Two triples with same subject name+type must share one canonical node id"
        )

    def test_same_name_different_type_stays_separate(self):
        """
        Device('SensorA') and Component('SensorA') must NOT be merged —
        type is part of the dedup key.
        """
        t1 = _make_triple("SensorA", "Device", "part_of", "Unit", "Component")
        t2 = _make_triple("SensorA", "Component", "part_of", "Unit", "Component")
        result = deduplicate_triples([t1, t2])

        subject_ids = {t.subject.id for t in result}
        assert len(subject_ids) == 2, (
            "Nodes with same name but different type must remain separate"
        )

    def test_merged_node_carries_union_of_source_docs(self):
        """
        Merged canonical node must hold source_docs from all originals
        (comma-separated union per spec dedup rule).
        """
        t1 = _make_triple("SensorA", "Device", "part_of", "Unit", "Component", source_doc="doc1.txt")
        t2 = _make_triple("SensorA", "Device", "causes", "Error42", "Error", source_doc="doc2.txt")
        result = deduplicate_triples([t1, t2])

        merged_sources = set()
        for t in result:
            # source_doc may be comma-separated or a list; handle both
            if isinstance(t.subject.source_doc, list):
                merged_sources.update(t.subject.source_doc)
            else:
                for s in t.subject.source_doc.split(","):
                    merged_sources.add(s.strip())

        assert "doc1.txt" in merged_sources
        assert "doc2.txt" in merged_sources


class TestDeduplicateEdges:
    """Edge dedup rules — keep highest weight when same (subj, pred, obj) appears twice."""

    def test_duplicate_triple_keeps_higher_weight(self):
        """
        When the same (subject name+type, predicate, object name+type) appears twice,
        only the higher-weight triple is kept.
        """
        low = _make_triple("SensorA", "Device", "part_of", "Unit", "Component", weight=0.5, source_doc="doc1.txt")
        high = _make_triple("SensorA", "Device", "part_of", "Unit", "Component", weight=0.9, source_doc="doc2.txt")
        result = deduplicate_triples([low, high])

        # Should collapse to one triple
        spo = [(t.subject.id, t.predicate, t.object.id) for t in result]
        unique_spo = set(spo)
        assert len(unique_spo) == 1, "Duplicate (s, p, o) must collapse to one triple"

        surviving = result[0]
        assert surviving.weight == pytest.approx(0.9), "Higher-weight triple must survive"

    def test_different_predicates_both_kept(self):
        """Two triples sharing subject+object but different predicates are NOT duplicates."""
        t1 = _make_triple("SensorA", "Device", "part_of", "Unit", "Component", weight=0.8)
        t2 = _make_triple("SensorA", "Device", "related_to", "Unit", "Component", weight=0.7)
        result = deduplicate_triples([t1, t2])
        predicates = {t.predicate for t in result}
        assert predicates == {"part_of", "related_to"}
