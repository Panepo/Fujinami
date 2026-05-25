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
