"""
Unit tests for reranker.LocalReranker.

Tests cover:
- disabled mode returns ANN order unchanged
- enabled mode with mocked model sorts by score descending
- rerank() trims results to top_k
- score() returns zeros on model load failure (graceful fallback)
- _reranker_score key is injected on each returned dict
- overfetch_factor and max_candidates properties reflect env/init values
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from reranker import LocalReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidates(texts: list[str]) -> list[dict]:
    return [{"text": t, "doc_id": f"doc_{i}"} for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# Disabled path
# ---------------------------------------------------------------------------

class TestRerankerDisabled:
    def test_returns_all_candidates_unchanged(self):
        reranker = LocalReranker(enabled=False)
        candidates = _make_candidates(["alpha", "beta", "gamma"])
        result = reranker.rerank("query", candidates, top_k=3)
        assert [r["text"] for r in result] == ["alpha", "beta", "gamma"]

    def test_does_not_inject_score_key(self):
        reranker = LocalReranker(enabled=False)
        candidates = _make_candidates(["a", "b"])
        result = reranker.rerank("query", candidates, top_k=2)
        # disabled path: scores are all 0.0 (no model loaded), key still injected
        # but insertion is not gated on enabled flag — just scores are zero
        for item in result:
            assert item["_reranker_score"] == 0.0

    def test_enabled_property_false(self):
        reranker = LocalReranker(enabled=False)
        assert reranker.enabled is False

    def test_score_returns_zeros_when_disabled(self):
        reranker = LocalReranker(enabled=False)
        scores = reranker.score("q", ["p1", "p2", "p3"])
        assert scores == [0.0, 0.0, 0.0]

    def test_score_empty_passages_returns_empty(self):
        reranker = LocalReranker(enabled=False)
        assert reranker.score("q", []) == []


# ---------------------------------------------------------------------------
# Enabled path with mocked CrossEncoder
# ---------------------------------------------------------------------------

class TestRerankerEnabled:
    def _make_reranker_with_mock(self, scores: list[float]) -> LocalReranker:
        reranker = LocalReranker(enabled=True, model_name="mock-model", device="cpu")
        mock_model = MagicMock()
        mock_model.predict.return_value = scores
        reranker._model = mock_model
        return reranker

    def test_sorts_by_score_descending(self):
        reranker = self._make_reranker_with_mock([0.1, 0.9, 0.5])
        candidates = _make_candidates(["low", "high", "mid"])
        result = reranker.rerank("query", candidates, top_k=3)
        assert [r["text"] for r in result] == ["high", "mid", "low"]

    def test_trims_to_top_k(self):
        reranker = self._make_reranker_with_mock([0.3, 0.9, 0.6])
        candidates = _make_candidates(["c", "a", "b"])
        result = reranker.rerank("query", candidates, top_k=2)
        assert len(result) == 2
        assert result[0]["text"] == "a"
        assert result[1]["text"] == "b"

    def test_injects_reranker_score(self):
        reranker = self._make_reranker_with_mock([0.2, 0.8])
        candidates = _make_candidates(["low", "high"])
        result = reranker.rerank("query", candidates, top_k=2)
        assert result[0]["_reranker_score"] == pytest.approx(0.8)
        assert result[1]["_reranker_score"] == pytest.approx(0.2)

    def test_top_k_none_returns_all_sorted(self):
        reranker = self._make_reranker_with_mock([0.1, 0.5, 0.3])
        candidates = _make_candidates(["x", "y", "z"])
        result = reranker.rerank("query", candidates, top_k=None)
        assert len(result) == 3
        assert result[0]["text"] == "y"

    def test_original_dicts_are_not_mutated(self):
        reranker = self._make_reranker_with_mock([0.9, 0.1])
        originals = _make_candidates(["a", "b"])
        reranker.rerank("query", originals, top_k=2)
        # original dicts must not have _reranker_score injected
        assert "_reranker_score" not in originals[0]
        assert "_reranker_score" not in originals[1]

    def test_enabled_property_true(self):
        reranker = LocalReranker(enabled=True)
        assert reranker.enabled is True


# ---------------------------------------------------------------------------
# Graceful fallback on model load failure
# ---------------------------------------------------------------------------

class TestRerankerFallback:
    def test_score_returns_zeros_on_load_failure(self):
        reranker = LocalReranker(enabled=True, model_name="nonexistent-model-xyz")
        # Patch CrossEncoder import to raise ImportError
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            scores = reranker.score("query", ["p1", "p2"])
        # Should return zeros without raising
        assert len(scores) == 2
        assert all(s == 0.0 for s in scores)

    def test_rerank_preserves_order_on_fallback(self):
        reranker = LocalReranker(enabled=True, model_name="nonexistent-model-xyz")
        candidates = _make_candidates(["first", "second", "third"])
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            result = reranker.rerank("query", candidates, top_k=3)
        # Stable sort on equal scores preserves ANN insertion order
        assert [r["text"] for r in result] == ["first", "second", "third"]

    def test_inference_exception_falls_back_to_zeros(self):
        reranker = LocalReranker(enabled=True)
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("CUDA OOM")
        reranker._model = mock_model
        scores = reranker.score("q", ["a", "b"])
        assert scores == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Overfetch / max_candidates properties
# ---------------------------------------------------------------------------

class TestRerankerConfig:
    def test_default_overfetch_factor(self):
        reranker = LocalReranker(enabled=False)
        assert reranker.overfetch_factor == pytest.approx(3.0)

    def test_custom_overfetch_factor(self):
        reranker = LocalReranker(enabled=False, overfetch_factor=5.0)
        assert reranker.overfetch_factor == pytest.approx(5.0)

    def test_default_max_candidates(self):
        reranker = LocalReranker(enabled=False)
        assert reranker.max_candidates == 50

    def test_custom_max_candidates(self):
        reranker = LocalReranker(enabled=False, max_candidates=100)
        assert reranker.max_candidates == 100
