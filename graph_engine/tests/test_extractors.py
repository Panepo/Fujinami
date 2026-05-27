"""
Tests for extractor public interface contract.

Rules derive from:
  - graph-design-spec.md §2 (Triple structure)
  - graph-design-spec.md §7 (pipeline: extraction stage)
  - graph-generation-analysis.md (extractor contract: extract(text, source_doc) -> list[Triple])

LLM expected JSON format (from analysis doc):
  [{"subject": "...", "subject_type": "...", "object": "...",
    "object_type": "...", "relation": "...", "confidence": 0.9}]

LLM calls are stubbed via urllib.request.urlopen — no real Ollama/network required.
spaCy extractor requires `en_core_web_sm` to be installed.
No implementation code was read.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
from graph_engine.models import Triple
from graph_engine.extractors.spacy_extractor import SpacyExtractor
from graph_engine.extractors.llm_extractor import LLMExtractor
from graph_engine.extractors.hybrid_extractor import HybridExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_extractor_contract(extractor_instance, text, source_doc="doc1.txt"):
    """Helper: verify an extractor returns list[Triple] where all carry source_doc."""
    result = extractor_instance.extract(text, source_doc)
    assert isinstance(result, list), "extract() must return a list"
    for triple in result:
        assert isinstance(triple, Triple), "Each element must be a Triple"
        assert triple.source_doc == source_doc, (
            f"Triple must carry source_doc='{source_doc}', got '{triple.source_doc}'"
        )
    return result


def _mock_urllib_response(content_str: str):
    """
    Return a context-manager mock that urllib.request.urlopen returns.
    Ollama response: {"message": {"content": "<content_str>"}}
    """
    body = json.dumps({"message": {"content": content_str}}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Valid LLM JSON — flat array per analysis doc format
# ---------------------------------------------------------------------------

_VALID_LLM_CONTENT = json.dumps([
    {
        "subject": "SensorA",
        "subject_type": "Device",
        "object": "ControlUnit",
        "object_type": "Component",
        "relation": "part_of",
        "confidence": 0.9,
    }
])

_MALFORMED_LLM_CONTENT = "Sure! Here are the triples: SensorA is part of ControlUnit okay?"


# ---------------------------------------------------------------------------
# spaCy extractor
# ---------------------------------------------------------------------------

class TestSpacyExtractor:

    @pytest.fixture
    def extractor(self):
        try:
            return SpacyExtractor()
        except OSError as e:
            pytest.skip(f"spaCy model not installed: {e}")

    def test_empty_text_returns_empty_list(self, extractor):
        result = extractor.extract("", "doc1.txt")
        assert result == []

    def test_returns_list_of_triples(self, extractor):
        text = "The SensorA device communicates with the ControlUnit via RS-485 protocol."
        _assert_extractor_contract(extractor, text, "spec_sheet.pdf")

    def test_all_triples_carry_source_doc(self, extractor):
        text = "IBM manufactures the Watson system in New York."
        result = extractor.extract(text, "manual.txt")
        for triple in result:
            assert triple.source_doc == "manual.txt"

    def test_whitespace_only_returns_empty_list(self, extractor):
        result = extractor.extract("   \n\t  ", "doc1.txt")
        assert result == []


# ---------------------------------------------------------------------------
# LLM extractor — stubs urllib.request.urlopen
# ---------------------------------------------------------------------------

class TestLLMExtractor:

    @pytest.fixture
    def extractor(self):
        return LLMExtractor(
            ollama_url="http://localhost:11434",
            model="test-model",
        )

    def test_empty_text_returns_empty_list(self, extractor):
        result = extractor.extract("", "doc1.txt")
        assert result == []

    def test_valid_json_response_produces_triples(self, extractor):
        """
        When Ollama returns valid JSON array, the extractor
        must parse it and return Triple objects.
        """
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _mock_urllib_response(_VALID_LLM_CONTENT)
            result = extractor.extract(
                "SensorA is part of ControlUnit.", "doc1.txt"
            )

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Triple)

    def test_valid_response_triples_carry_source_doc(self, extractor):
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _mock_urllib_response(_VALID_LLM_CONTENT)
            result = extractor.extract("SensorA is part of ControlUnit.", "manual.pdf")

        for triple in result:
            assert triple.source_doc == "manual.pdf"

    def test_malformed_json_returns_empty_list_without_crash(self, extractor):
        """
        When the LLM returns garbled / non-JSON text, the extractor
        must return an empty list and not raise an exception.
        """
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _mock_urllib_response(_MALFORMED_LLM_CONTENT)
            result = extractor.extract("SensorA is part of ControlUnit.", "doc1.txt")

        assert result == []

    def test_empty_array_in_response_returns_empty(self, extractor):
        """LLM returning '[]' must produce an empty list, not an error."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _mock_urllib_response("[]")
            result = extractor.extract("Some text.", "doc1.txt")

        assert result == []

    def test_ollama_connection_error_returns_empty_list(self, extractor):
        """Network failure must be handled gracefully — return [] not raise."""
        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            result = extractor.extract("Some text.", "doc1.txt")

        assert result == []


# ---------------------------------------------------------------------------
# Hybrid extractor — spaCy NER + LLM relation classification
# ---------------------------------------------------------------------------

# Text that produces spaCy dependency triples (subject–verb–object structure)
# en_core_web_sm reliably finds: Google (ORG), YouTube (ORG), Apple (ORG), Beats (ORG/PRODUCT)
# and dep-parsed triples: (Google, acquired, YouTube), (Apple, bought, Beats)
_HYBRID_TEXT = "Google acquired YouTube. Apple bought Beats by Dre."
_SPACY_KNOWN_NAMES = {"Google", "YouTube", "Apple", "Beats", "Beats by Dre"}

# LLM deliberately returns lowercase / wrong type — should be overridden by node_lookup
_HYBRID_LLM_RESPONSE = json.dumps([
    {
        "subject": "google",       # lowercase — spaCy has "Google"
        "subject_type": "Other",   # wrong type — spaCy has "Organization"
        "object": "youtube",       # lowercase — spaCy has "YouTube"
        "object_type": "Other",    # wrong type — spaCy has "Organization"
        "relation": "related_to",
        "confidence": 0.8,
    }
])


class TestHybridExtractor:

    @pytest.fixture
    def extractor(self):
        try:
            return HybridExtractor(
                ollama_url="http://localhost:11434",
                model="test-model",
                fallback_threshold=2,
            )
        except OSError as e:
            pytest.skip(f"spaCy model not installed: {e}")

    def test_node_names_come_from_spacy_not_llm(self, extractor):
        """
        Node names in triples must match spaCy's extraction (proper case),
        not the LLM's potentially mangled/lowercased names.
        """
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _mock_urllib_response(_HYBRID_LLM_RESPONSE)
            result = extractor.extract(_HYBRID_TEXT, "doc1.txt")

        assert len(result) >= 1, "Expected at least one triple from hybrid extraction"
        for triple in result:
            assert triple.subject.name in _SPACY_KNOWN_NAMES, (
                f"subject.name '{triple.subject.name}' should be spaCy's entity, not LLM's"
            )
            assert triple.object.name in _SPACY_KNOWN_NAMES, (
                f"object.name '{triple.object.name}' should be spaCy's entity, not LLM's"
            )

    def test_node_types_come_from_spacy_not_llm(self, extractor):
        """
        Node types must use spaCy's NER mapping, not the LLM's reclassified type.
        LLM returns 'Other' for Apple/Microsoft — spaCy maps ORG → 'Organization'.
        """
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _mock_urllib_response(_HYBRID_LLM_RESPONSE)
            result = extractor.extract(_HYBRID_TEXT, "doc1.txt")

        assert len(result) >= 1
        for triple in result:
            assert triple.subject.type != "Other", (
                f"subject.type should not be 'Other' — LLM's type leaked into node"
            )
            assert triple.object.type != "Other", (
                f"object.type should not be 'Other' — LLM's type leaked into node"
            )
