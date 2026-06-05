"""
Tests for entity type and relation type constraints.

Rules derive from:
  - graph-design-spec.md §3  (Entity Types — Strategy B, closed list)
  - graph-design-spec.md §4  (Relation Types — fully closed set)
  - Decisions log §8 (2026-05-20): free-text entity types not allowed; LLM must use Other.
  - Decisions log §8 (2026-05-20): fully closed relation set.

ENTITY_TYPES and RELATION_TYPES are typing.Literal aliases;
use typing.get_args() to extract the string values.
No implementation code was read.
"""

import typing
from graph_engine.models import ENTITY_TYPES, RELATION_TYPES
from graph_engine.state import ExtractionState, QueryState

# Extract string values from the Literal type aliases
_ENTITY_TYPE_VALUES = typing.get_args(ENTITY_TYPES)
_RELATION_TYPE_VALUES = typing.get_args(RELATION_TYPES)

# ---------------------------------------------------------------------------
# Authoritative lists from spec §3 and §4
# ---------------------------------------------------------------------------

_EXPECTED_ENTITY_TYPES = {
    "Person", "Organization", "Location", "Date", "Event", "Concept",
    "Device", "Component", "Specification", "Protocol", "Interface",
    "Software", "Standard", "Error", "Configuration", "Product",
    "Version", "Vendor",
    "Other",
}

_EXPECTED_RELATION_TYPES = {
    "is_a", "part_of", "causes", "governs", "opposes",
    "related_to", "supports_protocol", "certified_by",
}


class TestEntityTypes:
    """ENTITY_TYPES must exactly match the spec §3 standard list."""

    def test_entity_types_is_defined(self):
        assert ENTITY_TYPES is not None, "ENTITY_TYPES must be exported from graph_engine.models"

    def test_entity_types_has_string_values(self):
        assert len(_ENTITY_TYPE_VALUES) > 0, "ENTITY_TYPES must have at least one value"
        for v in _ENTITY_TYPE_VALUES:
            assert isinstance(v, str), f"All entity type values must be strings, got {type(v)}"

    def test_all_general_types_present(self):
        general = {"Person", "Organization", "Location", "Date", "Event", "Concept"}
        actual = set(_ENTITY_TYPE_VALUES)
        assert general.issubset(actual), f"Missing general types: {general - actual}"

    def test_all_technical_types_present(self):
        technical = {
            "Device", "Component", "Specification", "Protocol", "Interface",
            "Software", "Standard", "Error", "Configuration", "Product",
            "Version", "Vendor",
        }
        actual = set(_ENTITY_TYPE_VALUES)
        assert technical.issubset(actual), f"Missing technical types: {technical - actual}"

    def test_other_fallback_present(self):
        """spec §3 — 'Other' is the fallback when no type matches (decisions log 2026-05-20)."""
        assert "Other" in _ENTITY_TYPE_VALUES

    def test_no_undocumented_types(self):
        actual = set(_ENTITY_TYPE_VALUES)
        extra = actual - _EXPECTED_ENTITY_TYPES
        assert not extra, f"Undocumented entity types found: {extra}"

    def test_total_type_count(self):
        assert len(_ENTITY_TYPE_VALUES) == len(_EXPECTED_ENTITY_TYPES), (
            f"Expected {len(_EXPECTED_ENTITY_TYPES)} entity types, got {len(_ENTITY_TYPE_VALUES)}"
        )


class TestRelationTypes:
    """RELATION_TYPES must exactly match the spec §4 closed set."""

    def test_relation_types_is_defined(self):
        assert RELATION_TYPES is not None, "RELATION_TYPES must be exported from graph_engine.models"

    def test_relation_types_has_string_values(self):
        assert len(_RELATION_TYPE_VALUES) > 0
        for v in _RELATION_TYPE_VALUES:
            assert isinstance(v, str), f"All relation type values must be strings, got {type(v)}"

    def test_core_relations_present(self):
        core = {"is_a", "part_of", "causes", "governs", "opposes", "related_to"}
        actual = set(_RELATION_TYPE_VALUES)
        assert core.issubset(actual), f"Missing core relations: {core - actual}"

    def test_domain_relations_present(self):
        """spec §4 — domain-specific additions for device/spec docs."""
        domain = {"supports_protocol", "certified_by"}
        actual = set(_RELATION_TYPE_VALUES)
        assert domain.issubset(actual), f"Missing domain relations: {domain - actual}"

    def test_no_undocumented_relations(self):
        """spec §4 fully closed — no free-text relations allowed."""
        actual = set(_RELATION_TYPE_VALUES)
        extra = actual - _EXPECTED_RELATION_TYPES
        assert not extra, f"Undocumented relation types found: {extra}"

    def test_total_relation_count(self):
        assert len(_RELATION_TYPE_VALUES) == len(_EXPECTED_RELATION_TYPES), (
            f"Expected {len(_EXPECTED_RELATION_TYPES)} relation types, got {len(_RELATION_TYPE_VALUES)}"
        )


# ---------------------------------------------------------------------------
# ExtractionState and QueryState field validation
# ---------------------------------------------------------------------------

class TestExtractionState:
    """ExtractionState fields and partial initialization (total=False)."""

    def test_empty_dict_is_valid(self):
        """total=False allows empty dict as valid ExtractionState."""
        state: ExtractionState = {}
        assert isinstance(state, dict)

    def test_partial_init_with_required_fields(self):
        state: ExtractionState = {"raw_text": "hello", "source_doc": "doc.txt"}
        assert state["raw_text"] == "hello"
        assert state["source_doc"] == "doc.txt"

    def test_all_fields_accessible(self):
        state: ExtractionState = {
            "raw_text": "text",
            "source_doc": "doc.txt",
            "method": "spacy",
            "chunk_size": 200,
            "chunk_overlap": 20,
            "chunks": ["chunk1"],
            "triples": [],
            "deduped_triples": [],
            "stored_count": 0,
            "error": None,
        }
        assert state["method"] == "spacy"
        assert state["chunk_size"] == 200
        assert state["stored_count"] == 0
        assert state["error"] is None

    def test_chunks_field_accepts_list(self):
        state: ExtractionState = {"chunks": ["a", "b", "c"]}
        assert len(state["chunks"]) == 3

    def test_stored_count_field_accepts_int(self):
        state: ExtractionState = {"stored_count": 42}
        assert state["stored_count"] == 42


class TestQueryState:
    """QueryState fields and partial initialization (total=False)."""

    def test_empty_dict_is_valid(self):
        state: QueryState = {}
        assert isinstance(state, dict)

    def test_question_field(self):
        state: QueryState = {"question": "What is SensorA?"}
        assert state["question"] == "What is SensorA?"

    def test_all_fields_accessible(self):
        state: QueryState = {
            "question": "What is SensorA?",
            "method": "hybrid",
            "top_k": 5,
            "context": "Some context",
            "sources": [],
            "graphrag_context": "Graph info",
            "needs_graph": True,
            "answer": "SensorA is a device.",
            "iterations": 1,
            "node_trace": [],
        }
        assert state["method"] == "hybrid"
        assert state["needs_graph"] is True
        assert state["iterations"] == 1

    def test_node_trace_field_accepts_list(self):
        trace_entry = {"node": "vector_retrieve", "started_at": 0.0, "duration_ms": 42}
        state: QueryState = {"node_trace": [trace_entry]}
        assert len(state["node_trace"]) == 1
        assert state["node_trace"][0]["node"] == "vector_retrieve"

    def test_needs_graph_default_absent(self):
        """total=False — needs_graph need not be present."""
        state: QueryState = {"question": "test?"}
        assert "needs_graph" not in state

    def test_method_accepts_known_values(self):
        for method in ("vector", "graph", "hybrid"):
            state: QueryState = {"method": method}
            assert state["method"] == method

# Extract string values from the Literal type aliases
_ENTITY_TYPE_VALUES = typing.get_args(ENTITY_TYPES)
_RELATION_TYPE_VALUES = typing.get_args(RELATION_TYPES)

# ---------------------------------------------------------------------------
# Authoritative lists from spec §3 and §4
# ---------------------------------------------------------------------------

_EXPECTED_ENTITY_TYPES = {
    "Person", "Organization", "Location", "Date", "Event", "Concept",
    "Device", "Component", "Specification", "Protocol", "Interface",
    "Software", "Standard", "Error", "Configuration", "Product",
    "Version", "Vendor",
    "Other",
}

_EXPECTED_RELATION_TYPES = {
    "is_a", "part_of", "causes", "governs", "opposes",
    "related_to", "supports_protocol", "certified_by",
}


class TestEntityTypes:
    """ENTITY_TYPES must exactly match the spec §3 standard list."""

    def test_entity_types_is_defined(self):
        assert ENTITY_TYPES is not None, "ENTITY_TYPES must be exported from graph_engine.models"

    def test_entity_types_has_string_values(self):
        assert len(_ENTITY_TYPE_VALUES) > 0, "ENTITY_TYPES must have at least one value"
        for v in _ENTITY_TYPE_VALUES:
            assert isinstance(v, str), f"All entity type values must be strings, got {type(v)}"

    def test_all_general_types_present(self):
        general = {"Person", "Organization", "Location", "Date", "Event", "Concept"}
        actual = set(_ENTITY_TYPE_VALUES)
        assert general.issubset(actual), f"Missing general types: {general - actual}"

    def test_all_technical_types_present(self):
        technical = {
            "Device", "Component", "Specification", "Protocol", "Interface",
            "Software", "Standard", "Error", "Configuration", "Product",
            "Version", "Vendor",
        }
        actual = set(_ENTITY_TYPE_VALUES)
        assert technical.issubset(actual), f"Missing technical types: {technical - actual}"

    def test_other_fallback_present(self):
        """spec §3 — 'Other' is the fallback when no type matches (decisions log 2026-05-20)."""
        assert "Other" in _ENTITY_TYPE_VALUES

    def test_no_undocumented_types(self):
        actual = set(_ENTITY_TYPE_VALUES)
        extra = actual - _EXPECTED_ENTITY_TYPES
        assert not extra, f"Undocumented entity types found: {extra}"

    def test_total_type_count(self):
        assert len(_ENTITY_TYPE_VALUES) == len(_EXPECTED_ENTITY_TYPES), (
            f"Expected {len(_EXPECTED_ENTITY_TYPES)} entity types, got {len(_ENTITY_TYPE_VALUES)}"
        )


class TestRelationTypes:
    """RELATION_TYPES must exactly match the spec §4 closed set."""

    def test_relation_types_is_defined(self):
        assert RELATION_TYPES is not None, "RELATION_TYPES must be exported from graph_engine.models"

    def test_relation_types_has_string_values(self):
        assert len(_RELATION_TYPE_VALUES) > 0
        for v in _RELATION_TYPE_VALUES:
            assert isinstance(v, str), f"All relation type values must be strings, got {type(v)}"

    def test_core_relations_present(self):
        core = {"is_a", "part_of", "causes", "governs", "opposes", "related_to"}
        actual = set(_RELATION_TYPE_VALUES)
        assert core.issubset(actual), f"Missing core relations: {core - actual}"

    def test_domain_relations_present(self):
        """spec §4 — domain-specific additions for device/spec docs."""
        domain = {"supports_protocol", "certified_by"}
        actual = set(_RELATION_TYPE_VALUES)
        assert domain.issubset(actual), f"Missing domain relations: {domain - actual}"

    def test_no_undocumented_relations(self):
        """spec §4 fully closed — no free-text relations allowed."""
        actual = set(_RELATION_TYPE_VALUES)
        extra = actual - _EXPECTED_RELATION_TYPES
        assert not extra, f"Undocumented relation types found: {extra}"

    def test_total_relation_count(self):
        assert len(_RELATION_TYPE_VALUES) == len(_EXPECTED_RELATION_TYPES), (
            f"Expected {len(_EXPECTED_RELATION_TYPES)} relation types, got {len(_RELATION_TYPE_VALUES)}"
        )
