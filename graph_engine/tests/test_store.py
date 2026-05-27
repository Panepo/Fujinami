"""
Tests for GraphStore contract.

Rules derive from:
  - graph-design-spec.md §5 (Triple export — retrieval strategy C):
      "Include source_doc on each result" → Yes
      "Include weight score on each edge" → Yes — must always include
  - graph-design-spec.md §5 Decision: Triple export only (flat list, filterable by source_doc)
  - GraphStore interface: add_triples(), get_triples(), delete_by_source(), count()

get_triples() returns list[dict] — each dict is a flat triple row.
Tests verify the dict has the required keys from spec §5.
No implementation code was read.
"""

import pytest
from graph_engine.models import Node, Triple
from graph_engine.store import LanceDBGraphStore


def _make_triple(
    subj_name="SensorA", subj_type="Device",
    pred="part_of",
    obj_name="Unit", obj_type="Component",
    weight=0.85, source_doc="doc1.txt",
):
    subject = Node(name=subj_name, type=subj_type, source_doc=source_doc)
    obj = Node(name=obj_name, type=obj_type, source_doc=source_doc)
    return Triple(
        subject=subject, predicate=pred, object=obj,
        weight=weight, source_doc=source_doc,
    )


@pytest.fixture()
def tmp_store(tmp_path):
    """LanceDBGraphStore backed by a fresh temp directory."""
    store = LanceDBGraphStore(str(tmp_path / "lancedb"))
    yield store


class TestGraphStoreAdd:

    def test_count_starts_at_zero(self, tmp_store):
        assert tmp_store.count() == 0

    def test_count_increases_after_add(self, tmp_store):
        tmp_store.add_triples([_make_triple()])
        assert tmp_store.count() == 1

    def test_count_correct_after_multiple_adds(self, tmp_store):
        tmp_store.add_triples([
            _make_triple(subj_name="SensorA", source_doc="doc1.txt"),
            _make_triple(subj_name="SensorB", source_doc="doc2.txt"),
        ])
        assert tmp_store.count() == 2


class TestGraphStoreRetrieve:
    """spec §5 — Triple export, flat dict rows filterable by source_doc."""

    def test_retrieve_by_source_doc(self, tmp_store):
        tmp_store.add_triples([
            _make_triple(subj_name="SensorA", source_doc="doc1.txt"),
            _make_triple(subj_name="SensorB", source_doc="doc2.txt"),
        ])
        results = tmp_store.get_triples(source_doc="doc1.txt")
        assert len(results) == 1

    def test_retrieve_returns_only_matching_source_doc(self, tmp_store):
        tmp_store.add_triples([
            _make_triple(subj_name="SensorA", source_doc="doc1.txt"),
            _make_triple(subj_name="SensorB", source_doc="doc2.txt"),
        ])
        results = tmp_store.get_triples(source_doc="doc2.txt")
        source_docs = {r["source_doc"] for r in results}
        assert source_docs == {"doc2.txt"}, "Must not return triples from other documents"

    def test_retrieve_nonexistent_source_doc_returns_empty(self, tmp_store):
        tmp_store.add_triples([_make_triple(source_doc="doc1.txt")])
        results = tmp_store.get_triples(source_doc="unknown.txt")
        assert results == []

    def test_retrieved_row_has_source_doc_key(self, tmp_store):
        """spec §5 — source_doc must always be present on retrieved rows."""
        tmp_store.add_triples([_make_triple(source_doc="manual.pdf")])
        results = tmp_store.get_triples(source_doc="manual.pdf")
        assert "source_doc" in results[0], "Row dict must contain 'source_doc' key"
        assert results[0]["source_doc"] == "manual.pdf"

    def test_retrieved_row_has_weight_key(self, tmp_store):
        """spec §5 — weight must always be present on retrieved rows."""
        tmp_store.add_triples([_make_triple(weight=0.77, source_doc="manual.pdf")])
        results = tmp_store.get_triples(source_doc="manual.pdf")
        assert "weight" in results[0], "Row dict must contain 'weight' key"
        assert results[0]["weight"] == pytest.approx(0.77)

    def test_retrieved_row_has_subject_and_object_info(self, tmp_store):
        """spec §5 — result rows must carry subject and object info."""
        tmp_store.add_triples([
            _make_triple(subj_name="Alpha", obj_name="Beta", source_doc="doc1.txt")
        ])
        results = tmp_store.get_triples(source_doc="doc1.txt")
        row = results[0]
        # Accept any key name that conveys subject name (subject_name, subject, etc.)
        subject_val = row.get("subject_name") or row.get("subject") or row.get("subject_node")
        object_val = row.get("object_name") or row.get("object") or row.get("object_node")
        assert subject_val is not None, f"No subject key found in row: {list(row.keys())}"
        assert object_val is not None, f"No object key found in row: {list(row.keys())}"


class TestGraphStoreDelete:
    """delete_by_source must remove only that document's triples."""

    def test_delete_removes_target_document_triples(self, tmp_store):
        tmp_store.add_triples([
            _make_triple(subj_name="SensorA", source_doc="doc1.txt"),
            _make_triple(subj_name="SensorB", source_doc="doc2.txt"),
        ])
        tmp_store.delete_by_source("doc1.txt")
        assert tmp_store.get_triples(source_doc="doc1.txt") == []

    def test_delete_keeps_other_document_triples(self, tmp_store):
        tmp_store.add_triples([
            _make_triple(subj_name="SensorA", source_doc="doc1.txt"),
            _make_triple(subj_name="SensorB", source_doc="doc2.txt"),
        ])
        tmp_store.delete_by_source("doc1.txt")
        results = tmp_store.get_triples(source_doc="doc2.txt")
        assert len(results) == 1, "doc2 triples must survive deletion of doc1"

    def test_count_decreases_after_delete(self, tmp_store):
        tmp_store.add_triples([
            _make_triple(subj_name="SensorA", source_doc="doc1.txt"),
            _make_triple(subj_name="SensorB", source_doc="doc2.txt"),
        ])
        assert tmp_store.count() == 2
        tmp_store.delete_by_source("doc1.txt")
        assert tmp_store.count() == 1

    def test_delete_nonexistent_source_does_not_raise(self, tmp_store):
        """Deleting a source_doc that was never stored must not raise."""
        tmp_store.delete_by_source("never_indexed.txt")


class TestMethodIsolation:
    """Triples from different extraction methods must not overwrite each other."""

    def test_add_triples_with_method_stores_method_field(self, tmp_store):
        """Triple stored with method='llm' must return method='llm' on read."""
        t = _make_triple(subj_name="Alpha", source_doc="doc1.txt")
        t.method = "llm"
        tmp_store.add_triples([t])
        results = tmp_store.get_triples(source_doc="doc1.txt")
        assert len(results) == 1
        assert results[0]["method"] == "llm"

    def test_same_doc_different_methods_both_kept(self, tmp_store):
        """Same doc indexed with spacy AND llm → both sets of triples kept."""
        t1 = _make_triple(subj_name="Alpha", obj_name="Beta", source_doc="doc1.txt")
        t1.method = "spacy"
        t2 = _make_triple(subj_name="Alpha", obj_name="Beta", source_doc="doc1.txt")
        t2.method = "llm"
        tmp_store.add_triples([t1])
        tmp_store.add_triples([t2])
        assert tmp_store.count() == 2

    def test_delete_by_source_and_method_only_removes_target(self, tmp_store):
        """delete_by_source_and_method(doc1, spacy) must keep doc1/llm triples."""
        t1 = _make_triple(subj_name="Alpha", source_doc="doc1.txt")
        t1.method = "spacy"
        t2 = _make_triple(subj_name="Alpha", source_doc="doc1.txt")
        t2.method = "llm"
        tmp_store.add_triples([t1])
        tmp_store.add_triples([t2])
        tmp_store.delete_by_source_and_method("doc1.txt", "spacy")
        results = tmp_store.get_triples(source_doc="doc1.txt")
        assert len(results) == 1
        assert results[0]["method"] == "llm"

    def test_delete_by_source_removes_all_methods(self, tmp_store):
        """Old delete_by_source still removes ALL methods for a doc."""
        t1 = _make_triple(subj_name="Alpha", source_doc="doc1.txt")
        t1.method = "spacy"
        t2 = _make_triple(subj_name="Alpha", source_doc="doc1.txt")
        t2.method = "llm"
        tmp_store.add_triples([t1])
        tmp_store.add_triples([t2])
        tmp_store.delete_by_source("doc1.txt")
        assert tmp_store.get_triples(source_doc="doc1.txt") == []

    def test_get_triples_filter_by_method(self, tmp_store):
        """get_triples(method='spacy') returns only spacy triples."""
        t1 = _make_triple(subj_name="Alpha", source_doc="doc1.txt")
        t1.method = "spacy"
        t2 = _make_triple(subj_name="Beta", source_doc="doc1.txt")
        t2.method = "llm"
        tmp_store.add_triples([t1, t2])
        results = tmp_store.get_triples(method="spacy")
        assert len(results) == 1
        assert results[0]["subject"]["name"] == "Alpha"

    def test_get_triples_no_method_filter_returns_all(self, tmp_store):
        """Default get_triples() returns triples from all methods."""
        t1 = _make_triple(subj_name="Alpha", source_doc="doc1.txt")
        t1.method = "spacy"
        t2 = _make_triple(subj_name="Beta", source_doc="doc1.txt")
        t2.method = "llm"
        tmp_store.add_triples([t1, t2])
        results = tmp_store.get_triples(source_doc="doc1.txt")
        assert len(results) == 2
