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
