"""
Local reranker module for post-retrieval passage ordering.

This module is intentionally standalone so retriever.py can consume a stable
interface without carrying model initialization logic.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_ENABLE_RERANKER = os.environ.get("ENABLE_RERANKER", "false").lower() in (
    "1",
    "true",
    "yes",
)
_RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
_RERANKER_DEVICE = os.environ.get("RERANKER_DEVICE", "auto")
_RERANKER_BATCH_SIZE = int(os.environ.get("RERANKER_BATCH_SIZE", "16"))
_RERANKER_OVERFETCH_FACTOR = float(os.environ.get("RERANKER_OVERFETCH_FACTOR", "3.0"))
_RERANKER_MAX_CANDIDATES = int(os.environ.get("RERANKER_MAX_CANDIDATES", "50"))


@dataclass(slots=True)
class RerankItem:
    """Container for reranked output entries."""

    score: float
    payload: dict[str, Any]


class LocalReranker:
    """
    Local cross-encoder reranker with lazy model loading.

    Expected usage:
    1. Instantiate once per process or per retriever instance.
    2. Call rerank(query, passages, top_k) to obtain sorted candidates.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        enabled: bool | None = None,
        overfetch_factor: float | None = None,
        max_candidates: int | None = None,
    ) -> None:
        self._enabled = _ENABLE_RERANKER if enabled is None else enabled
        self._model_name = model_name or _RERANKER_MODEL
        self._device = device or _RERANKER_DEVICE
        self._batch_size = batch_size if batch_size is not None else _RERANKER_BATCH_SIZE
        self._overfetch_factor = overfetch_factor if overfetch_factor is not None else _RERANKER_OVERFETCH_FACTOR
        self._max_candidates = max_candidates if max_candidates is not None else _RERANKER_MAX_CANDIDATES

        self._model: Any | None = None

    @property
    def enabled(self) -> bool:
        """Return whether reranking is enabled by configuration."""
        return self._enabled

    @property
    def is_ready(self) -> bool:
        """Return True when model is loaded and ready for inference."""
        return self._model is not None

    @property
    def overfetch_factor(self) -> float:
        """ANN candidate multiplier used by the retriever when reranker is active."""
        return self._overfetch_factor

    @property
    def max_candidates(self) -> int:
        """Hard ceiling on ANN candidates fetched for reranking."""
        return self._max_candidates

    def _resolve_device(self) -> str:
        """Resolve runtime device with safe fallback to CPU."""
        if self._device != "auto":
            return self._device

        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            # Device probing should never break retrieval startup.
            pass

        return "cpu"

    def _ensure_loaded(self) -> bool:
        """Load model lazily; return False if unavailable."""
        if not self._enabled:
            return False
        if self._model is not None:
            return True
        # Do not cache load errors permanently — allow retry on the next request
        # so that a corrected environment (e.g. after dependency install) takes
        # effect without a full process restart.

        try:
            import inspect  # noqa: PLC0415

            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            device = self._resolve_device()

            # Newer transformers + accelerate versions load models with meta tensors
            # by default, which breaks CrossEncoder's internal .to(device) call.
            # Passing device_map=None via model_kwargs / automodel_args prevents this.
            _ce_params = inspect.signature(CrossEncoder.__init__).parameters
            if "model_kwargs" in _ce_params:
                # sentence-transformers >= 3.x
                self._model = CrossEncoder(
                    self._model_name,
                    device=device,
                    model_kwargs={"device_map": None},
                )
            elif "automodel_args" in _ce_params:
                # sentence-transformers 2.x
                self._model = CrossEncoder(
                    self._model_name,
                    device=device,
                    automodel_args={"device_map": None},
                )
            else:
                self._model = CrossEncoder(self._model_name, device=device)

            logger.info(
                "Reranker loaded: model=%s device=%s batch_size=%s",
                self._model_name,
                device,
                self._batch_size,
            )
            return True
        except Exception as exc:
            logger.warning("Reranker unavailable, fallback to ANN ordering: %s", exc)
            return False

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """
        Score passages against a query.

        Returns ANN-neutral scores (all zeros) when model cannot be loaded.
        """
        if not passages:
            return []

        if not self._ensure_loaded():
            return [0.0] * len(passages)

        pairs = [[query, p] for p in passages]

        try:
            scores = self._model.predict(pairs, batch_size=self._batch_size)
            return [float(s) for s in scores]
        except Exception as exc:
            logger.warning("Reranker inference failed, fallback to ANN ordering: %s", exc)
            return [0.0] * len(passages)

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_k: int | None = None,
        text_key: str = "text",
    ) -> list[dict[str, Any]]:
        """
        Return candidates sorted by reranker score descending.

        The returned dicts include an additional key: _reranker_score.
        """
        if not candidates:
            return []

        passages = [str(c.get(text_key, "")) for c in candidates]
        scores = self.score(query, passages)

        items = [
            RerankItem(score=scores[idx], payload=dict(candidates[idx]))
            for idx in range(len(candidates))
        ]

        # Stable sort preserves ANN order when scores are tied or all-zero fallback.
        items.sort(key=lambda item: item.score, reverse=True)

        if top_k is not None:
            items = items[:top_k]

        reranked: list[dict[str, Any]] = []
        for item in items:
            item.payload["_reranker_score"] = float(item.score)
            reranked.append(item.payload)

        return reranked
