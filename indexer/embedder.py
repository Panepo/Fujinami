"""Ollama HTTP embedder — direct POST to /api/embed, L2-normalised float32."""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
import numpy as np

from auth_utils import normalize_bearer_token

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Embeds texts via Ollama's ``/api/embed`` endpoint.

    Parameters
    ----------
    model:
        Ollama model name (e.g. ``"nomic-embed-text"``).
    ollama_base_url:
        Base URL of the Ollama server (e.g. ``"http://localhost:11434"``).
    dimension:
        Expected embedding dimension. Inferred from the first API response if
        ``None``.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        ollama_base_url: str,
        dimension: Optional[int] = None,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = ollama_base_url.rstrip("/")
        self._dimension = dimension
        self._timeout = timeout
        self._auth_header = normalize_bearer_token(os.environ.get("OLLAMA_BEARER", ""))

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed *texts* and return an L2-normalised float32 array of shape ``(n, dim)``.

        Parameters
        ----------
        texts:
            List of strings to embed.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), dimension)``, dtype ``float32``, L2-normalised.
        """
        if not texts:
            dim = self._dimension or 0
            return np.empty((0, dim), dtype=np.float32)

        response = httpx.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": texts},
            headers=(
                {"Authorization": self._auth_header}
                if self._auth_header
                else None
            ),
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = np.array(data["embeddings"], dtype=np.float32)

        if self._dimension is None:
            self._dimension = embeddings.shape[1]
            logger.info("OllamaEmbedder: inferred dimension=%d", self._dimension)

        # L2 normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (embeddings / norms).astype(np.float32)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> Optional[int]:
        return self._dimension
