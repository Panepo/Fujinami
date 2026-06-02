"""
Tests for the GraphPipeline end-to-end contract.

Rules derive from:
  - graph-design-spec.md §7 (Pipeline: Documents → Chunking → Extraction → Dedup → Store → API)
  - graph-generation-analysis.md §7 (pipeline.run() returns stats dict)
  - Agreed stats keys: chunks, raw_triples, deduplicated_triples, stored

Uses a stub extractor so no LLM/spaCy/network is required.
No implementation code was read.
"""

import pytest
from graph_engine.models import Node, Triple
from graph_engine.base import BaseExtractor
from graph_engine.pipeline import GraphPipeline
from graph_engine.store import LanceDBGraphStore
from graph_engine.state import ExtractionState


# ---------------------------------------------------------------------------
# Stub extractor — returns a fixed set of triples so tests are deterministic
# ---------------------------------------------------------------------------

class StubExtractor(BaseExtractor):
    """Returns a configurable list of triples; used to avoid real LLM/spaCy calls."""

    def __init__(self, triples: list[Triple] | None = None):
        self._triples = triples or []

    def extract(self, text: str, source_doc: str) -> list[Triple]:
        if not text.strip():
            return []
        # Re-stamp source_doc so they match what the pipeline passes
        result = []
        for t in self._triples:
            subject = Node(name=t.subject.name, type=t.subject.type, source_doc=source_doc)
            obj = Node(name=t.object.name, type=t.object.type, source_doc=source_doc)
            result.append(Triple(
                subject=subject, predicate=t.predicate, object=obj,
                weight=t.weight, source_doc=source_doc,
            ))
        return result


def _sample_triple(source_doc="doc1.txt") -> Triple:
    subject = Node(name="SensorA", type="Device", source_doc=source_doc)
    obj = Node(name="Unit", type="Component", source_doc=source_doc)
    return Triple(subject=subject, predicate="part_of", object=obj, weight=0.9, source_doc=source_doc)


@pytest.fixture
def store(tmp_path):
    return LanceDBGraphStore(str(tmp_path / "lancedb"))


@pytest.fixture
def pipeline_with_triples(store):
    extractor = StubExtractor(triples=[_sample_triple()])
    return GraphPipeline(extractor=extractor, store=store)


@pytest.fixture
def pipeline_empty(store):
    extractor = StubExtractor(triples=[])
    return GraphPipeline(extractor=extractor, store=store)


# ---------------------------------------------------------------------------
# Stats dict contract (from spec §7 pipeline)
# ---------------------------------------------------------------------------

class TestPipelineStatsDict:
    """pipeline.run() must return a dict with the agreed stat keys."""

    EXPECTED_KEYS = {"chunks", "raw_triples", "deduplicated_triples", "stored"}

    def test_run_returns_dict(self, pipeline_with_triples):
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert isinstance(result, dict)

    def test_run_stats_has_all_required_keys(self, pipeline_with_triples):
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        missing = self.EXPECTED_KEYS - result.keys()
        assert not missing, f"Stats dict missing keys: {missing}"

    def test_chunks_is_non_negative_integer(self, pipeline_with_triples):
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert isinstance(result["chunks"], int) and result["chunks"] >= 0

    def test_raw_triples_is_non_negative_integer(self, pipeline_with_triples):
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert isinstance(result["raw_triples"], int) and result["raw_triples"] >= 0

    def test_deduplicated_triples_is_non_negative_integer(self, pipeline_with_triples):
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert isinstance(result["deduplicated_triples"], int) and result["deduplicated_triples"] >= 0

    def test_stored_is_non_negative_integer(self, pipeline_with_triples):
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert isinstance(result["stored"], int) and result["stored"] >= 0

    def test_deduplicated_lte_raw(self, pipeline_with_triples):
        """After dedup, stored count must be ≤ raw extracted count."""
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert result["deduplicated_triples"] <= result["raw_triples"]

    def test_stored_equals_deduplicated(self, pipeline_with_triples):
        """Everything that survives dedup must be stored."""
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert result["stored"] == result["deduplicated_triples"]


# ---------------------------------------------------------------------------
# Empty text — spec §7 pipeline should handle gracefully
# ---------------------------------------------------------------------------

class TestPipelineEmptyText:

    def test_empty_text_returns_zero_stored(self, pipeline_empty):
        result = pipeline_empty.run("", "doc1.txt")
        assert result["stored"] == 0

    def test_empty_text_does_not_raise(self, pipeline_empty):
        pipeline_empty.run("", "doc1.txt")  # must not throw

    def test_whitespace_text_returns_zero_stored(self, pipeline_empty):
        result = pipeline_empty.run("   \n\t  ", "doc1.txt")
        assert result["stored"] == 0

    def test_empty_text_chunks_zero(self, pipeline_empty):
        result = pipeline_empty.run("", "doc1.txt")
        assert result["chunks"] == 0


# ---------------------------------------------------------------------------
# Pipeline → Store integration
# ---------------------------------------------------------------------------

class TestPipelineStoreIntegration:
    """Triples produced by the pipeline must actually end up in the store."""

    def test_stored_count_matches_store_count(self, pipeline_with_triples, store):
        pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert store.count() == 1

    def test_stored_triples_retrievable_by_source_doc(self, pipeline_with_triples, store):
        pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        results = store.get_triples(source_doc="doc1.txt")
        assert len(results) == 1

    def test_delete_source_removes_pipeline_triples(self, pipeline_with_triples, store):
        pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert store.count() == 1

        pipeline_with_triples.delete_source("doc1.txt")
        assert store.count() == 0


# ---------------------------------------------------------------------------
# ExtractionGraph via ExtractionState (LangGraph-based pipeline)
# ---------------------------------------------------------------------------

class TestExtractionGraphState:
    """ExtractionGraph.invoke(ExtractionState) must propagate state correctly through all nodes."""

    def test_invoke_returns_extraction_state(self, store):
        from graph_engine.pipeline import ExtractionGraph
        from graph_engine.extractors.spacy_extractor import SpacyExtractor
        extractor = StubExtractor(triples=[_sample_triple()])
        eg = ExtractionGraph(extractor=extractor, store=store)
        state: ExtractionState = {
            "raw_text": "SensorA is part of Unit.",
            "source_doc": "doc1.txt",
            "method": "spacy",
            "chunk_size": 200,
            "chunk_overlap": 20,
        }
        result = eg.invoke(state)
        assert isinstance(result, dict)

    def test_invoke_populates_chunks(self, store):
        from graph_engine.pipeline import ExtractionGraph
        extractor = StubExtractor(triples=[_sample_triple()])
        eg = ExtractionGraph(extractor=extractor, store=store)
        state: ExtractionState = {
            "raw_text": "SensorA is part of Unit.",
            "source_doc": "doc1.txt",
        }
        result = eg.invoke(state)
        assert "chunks" in result
        assert isinstance(result["chunks"], list)

    def test_invoke_populates_triples(self, store):
        from graph_engine.pipeline import ExtractionGraph
        extractor = StubExtractor(triples=[_sample_triple()])
        eg = ExtractionGraph(extractor=extractor, store=store)
        state: ExtractionState = {
            "raw_text": "SensorA is part of Unit.",
            "source_doc": "doc1.txt",
        }
        result = eg.invoke(state)
        assert "triples" in result
        assert isinstance(result["triples"], list)

    def test_invoke_populates_stored_count(self, store):
        from graph_engine.pipeline import ExtractionGraph
        extractor = StubExtractor(triples=[_sample_triple()])
        eg = ExtractionGraph(extractor=extractor, store=store)
        state: ExtractionState = {
            "raw_text": "SensorA is part of Unit.",
            "source_doc": "doc1.txt",
        }
        result = eg.invoke(state)
        assert "stored_count" in result
        assert result["stored_count"] >= 0

    def test_invoke_empty_text_stored_count_zero(self, store):
        from graph_engine.pipeline import ExtractionGraph
        extractor = StubExtractor(triples=[_sample_triple()])
        eg = ExtractionGraph(extractor=extractor, store=store)
        state: ExtractionState = {
            "raw_text": "",
            "source_doc": "doc1.txt",
        }
        result = eg.invoke(state)
        assert result.get("stored_count", 0) == 0

    def test_invoke_no_error_on_valid_input(self, store):
        from graph_engine.pipeline import ExtractionGraph
        extractor = StubExtractor(triples=[_sample_triple()])
        eg = ExtractionGraph(extractor=extractor, store=store)
        state: ExtractionState = {
            "raw_text": "SensorA is part of Unit.",
            "source_doc": "doc1.txt",
        }
        result = eg.invoke(state)
        assert "error" not in result or result["error"] is None


# ---------------------------------------------------------------------------
# Stub extractor — returns a fixed set of triples so tests are deterministic
# ---------------------------------------------------------------------------

class StubExtractor(BaseExtractor):
    """Returns a configurable list of triples; used to avoid real LLM/spaCy calls."""

    def __init__(self, triples: list[Triple] | None = None):
        self._triples = triples or []

    def extract(self, text: str, source_doc: str) -> list[Triple]:
        if not text.strip():
            return []
        # Re-stamp source_doc so they match what the pipeline passes
        result = []
        for t in self._triples:
            subject = Node(name=t.subject.name, type=t.subject.type, source_doc=source_doc)
            obj = Node(name=t.object.name, type=t.object.type, source_doc=source_doc)
            result.append(Triple(
                subject=subject, predicate=t.predicate, object=obj,
                weight=t.weight, source_doc=source_doc,
            ))
        return result


def _sample_triple(source_doc="doc1.txt") -> Triple:
    subject = Node(name="SensorA", type="Device", source_doc=source_doc)
    obj = Node(name="Unit", type="Component", source_doc=source_doc)
    return Triple(subject=subject, predicate="part_of", object=obj, weight=0.9, source_doc=source_doc)


@pytest.fixture
def store(tmp_path):
    return LanceDBGraphStore(str(tmp_path / "lancedb"))


@pytest.fixture
def pipeline_with_triples(store):
    extractor = StubExtractor(triples=[_sample_triple()])
    return GraphPipeline(extractor=extractor, store=store)


@pytest.fixture
def pipeline_empty(store):
    extractor = StubExtractor(triples=[])
    return GraphPipeline(extractor=extractor, store=store)


# ---------------------------------------------------------------------------
# Stats dict contract (from spec §7 pipeline)
# ---------------------------------------------------------------------------

class TestPipelineStatsDict:
    """pipeline.run() must return a dict with the agreed stat keys."""

    EXPECTED_KEYS = {"chunks", "raw_triples", "deduplicated_triples", "stored"}

    def test_run_returns_dict(self, pipeline_with_triples):
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert isinstance(result, dict)

    def test_run_stats_has_all_required_keys(self, pipeline_with_triples):
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        missing = self.EXPECTED_KEYS - result.keys()
        assert not missing, f"Stats dict missing keys: {missing}"

    def test_chunks_is_non_negative_integer(self, pipeline_with_triples):
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert isinstance(result["chunks"], int) and result["chunks"] >= 0

    def test_raw_triples_is_non_negative_integer(self, pipeline_with_triples):
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert isinstance(result["raw_triples"], int) and result["raw_triples"] >= 0

    def test_deduplicated_triples_is_non_negative_integer(self, pipeline_with_triples):
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert isinstance(result["deduplicated_triples"], int) and result["deduplicated_triples"] >= 0

    def test_stored_is_non_negative_integer(self, pipeline_with_triples):
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert isinstance(result["stored"], int) and result["stored"] >= 0

    def test_deduplicated_lte_raw(self, pipeline_with_triples):
        """After dedup, stored count must be ≤ raw extracted count."""
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert result["deduplicated_triples"] <= result["raw_triples"]

    def test_stored_equals_deduplicated(self, pipeline_with_triples):
        """Everything that survives dedup must be stored."""
        result = pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert result["stored"] == result["deduplicated_triples"]


# ---------------------------------------------------------------------------
# Empty text — spec §7 pipeline should handle gracefully
# ---------------------------------------------------------------------------

class TestPipelineEmptyText:

    def test_empty_text_returns_zero_stored(self, pipeline_empty):
        result = pipeline_empty.run("", "doc1.txt")
        assert result["stored"] == 0

    def test_empty_text_does_not_raise(self, pipeline_empty):
        pipeline_empty.run("", "doc1.txt")  # must not throw

    def test_whitespace_text_returns_zero_stored(self, pipeline_empty):
        result = pipeline_empty.run("   \n\t  ", "doc1.txt")
        assert result["stored"] == 0

    def test_empty_text_chunks_zero(self, pipeline_empty):
        result = pipeline_empty.run("", "doc1.txt")
        assert result["chunks"] == 0


# ---------------------------------------------------------------------------
# Pipeline → Store integration
# ---------------------------------------------------------------------------

class TestPipelineStoreIntegration:
    """Triples produced by the pipeline must actually end up in the store."""

    def test_stored_count_matches_store_count(self, pipeline_with_triples, store):
        pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert store.count() == 1

    def test_stored_triples_retrievable_by_source_doc(self, pipeline_with_triples, store):
        pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        results = store.get_triples(source_doc="doc1.txt")
        assert len(results) == 1

    def test_delete_source_removes_pipeline_triples(self, pipeline_with_triples, store):
        pipeline_with_triples.run("SensorA is part of Unit.", "doc1.txt")
        assert store.count() == 1

        pipeline_with_triples.delete_source("doc1.txt")
        assert store.count() == 0
