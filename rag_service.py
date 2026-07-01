"""
RagService â€” thin faÃ§ade over RagIndexer + RagRetriever.

Stack
-----
- Indexing LLM / VLM / Embeddings  : Ollama on OLLAMA_INDEX_URL
- Chat LLM                         : Ollama on OLLAMA_CHAT_URL
- Vector store                     : LanceDB (persistent, embedded, file-based)
- Graph engine                     : graph_engine (local triple extraction + LanceDB storage)

This module delegates all work to:
  :class:`indexer.RagIndexer`   â€” document loading, delta detection, LanceDB upsert, graph extraction
  :class:`retriever.RagRetriever` â€” vector search, graph context, response generation
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from indexer import RagIndexer, SUPPORTED_EXTENSIONS
from retriever import RagRetriever

logger = logging.getLogger(__name__)


class RagService:
    """
    Hybrid retrieval-augmented generation service.

    Thin faÃ§ade over :class:`indexer.RagIndexer` and :class:`retriever.RagRetriever`.

    Parameters
    ----------
    collection_name:
        Name of the document collection (e.g. ``"harusame"``).
        When set, documents live in ``{root_dir}/data/{collection_name}/``
        and each collection gets its own isolated ragdata and LanceDB store
        under ``{root_dir}/ragdata/{collection_name}/``.
        When ``None`` (default), the legacy single-collection layout is used.
    root_dir:
        Root directory for data, ragdata, and LanceDB storage.
        Defaults to the directory containing this file.
    lance_db_path:
        Path to the LanceDB database directory.
        Defaults to ``{ragdata_dir}/lancedb``.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        root_dir: str | Path | None = None,
        lance_db_path: str | Path | None = None,
    ) -> None:
        kwargs = dict(
            collection_name=collection_name,
            root_dir=root_dir,
            lance_db_path=lance_db_path,
        )
        self._indexer = RagIndexer(**kwargs)
        self._retriever = RagRetriever(**kwargs)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def indexer(self) -> RagIndexer:
        return self._indexer

    @property
    def retriever(self) -> RagRetriever:
        return self._retriever

    # ------------------------------------------------------------------
    # Indexing (delegated to RagIndexer)
    # ------------------------------------------------------------------

    async def index_documents(
        self,
        documents_dir: str | Path | None = None,
        mode: str = "all",
        force: bool = False,
    ) -> None:
        """Incremental indexing pipeline. Delegates to :class:`RagIndexer`."""
        await self._indexer.index_documents(
            documents_dir=documents_dir,
            mode=mode,
            force=force,
        )
        # Reopen the retriever's LanceDB table reference so newly indexed
        # chunks are immediately visible to subsequent queries.
        self._retriever.reload_table()

    # ------------------------------------------------------------------
    # Search (delegated to RagRetriever)
    # ------------------------------------------------------------------

    async def vector_search(self, query: str, top_k: int | None = None) -> str:
        """Pure semantic similarity search. Delegates to :class:`RagRetriever`."""
        kwargs = {} if top_k is None else {"top_k": top_k}
        return await self._retriever.vector_search(query, **kwargs)

    async def global_search(self, query: str) -> str:
        """Global knowledge-graph search. Delegates to :class:`RagRetriever`."""
        return await self._retriever.global_search(query)

    async def hybrid_search(self, query: str, top_k: int | None = None) -> str:
        """Hybrid vector + graph search. Delegates to :class:`RagRetriever`."""
        kwargs = {} if top_k is None else {"top_k": top_k}
        return await self._retriever.hybrid_search(query, **kwargs)

    def get_document_chunks(self, filename: str) -> list[dict]:
        """Return all stored chunks for *filename*. Delegates to :class:`RagRetriever`."""
        return self._retriever.get_document_chunks(filename)

    # ------------------------------------------------------------------
    # Internal helpers (delegated to RagRetriever, used by api.py)
    # ------------------------------------------------------------------

    async def _raw_vector_results(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return raw LanceDB rows for *query*. Delegates to :class:`RagRetriever`."""
        kwargs: dict = {} if top_k is None else {"top_k": top_k}
        return await self._retriever._raw_vector_results(query, **kwargs)

    async def _raw_vector_results_from_embedding(
        self, embedding: list[float], top_k: int | None = None
    ) -> list[dict]:
        """Return raw LanceDB rows for a pre-computed *embedding*. Delegates to :class:`RagRetriever`."""
        kwargs: dict = {} if top_k is None else {"top_k": top_k}
        return await self._retriever._raw_vector_results_from_embedding(embedding, **kwargs)

    async def _raw_vector_context(self, query: str, top_k: int | None = None) -> str:
        """Return concatenated text chunks for *query*. Delegates to :class:`RagRetriever`."""
        kwargs: dict = {} if top_k is None else {"top_k": top_k}
        return await self._retriever._raw_vector_context(query, **kwargs)

    async def _graphrag_search(self, query: str, method: str = "local") -> str:
        """Return graph context for *query*. Delegates to :class:`RagRetriever`."""
        return await asyncio.to_thread(self._retriever._graph_context, query)

    async def _generate_response(self, query: str, context: str, image_base64: str | None = None) -> str:
        """Generate a response for *query* given *context* and optional image. Delegates to :class:`RagRetriever`."""
        return await self._retriever._generate_response(query, context, image_base64)

    @property
    def _chat_service(self):
        """Expose the underlying chat service. Delegates to :class:`RagRetriever`."""
        return self._retriever._chat_service


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def get_rag_service(
    collection_name: str,
    root_dir: str | Path | None = None,
) -> RagService:
    """Return a :class:`RagService` scoped to *collection_name*.

    Parameters
    ----------
    collection_name:
        Name of the document collection (e.g. ``"harusame"``).
    root_dir:
        Root directory for data and ragdata storage.
        Defaults to the directory containing this file.
    """
    return RagService(collection_name=collection_name, root_dir=root_dir)
