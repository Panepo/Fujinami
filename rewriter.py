"""
QueryRewriter — LLM-based query pre-processing for improved retrieval.

Supports three modes:
- HyDE (Hypothetical Document Embeddings): generate a hypothetical answer,
  embed it, use that vector for similarity search.
- multi_query: generate N alternative phrasings, run parallel searches, merge.
- step_back: broaden to a higher-level question, combine with original.
"""
from __future__ import annotations

import logging

from models import RewriteMeta

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Stateless query pre-processor.  Instantiated per-request in api.py,
    reusing the existing LLM and embedding service connections from RagRetriever.

    Parameters
    ----------
    llm:
        A ChatOllama instance (same object used by RagRetriever).
    embedding_service:
        An OllamaEmbeddings instance used for query-time embeddings.
    """

    def __init__(self, llm, embedding_service) -> None:
        self._llm = llm
        self._embedder = embedding_service

    # ------------------------------------------------------------------
    # HyDE
    # ------------------------------------------------------------------

    async def hyde(self, query: str) -> tuple[list[float], RewriteMeta]:
        """
        Generate a short hypothetical passage that would answer *query*, then
        embed that passage to use as the search vector.

        Returns
        -------
        (embedding, RewriteMeta)
        """
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415
        import asyncio  # noqa: PLC0415

        messages = [
            SystemMessage(content=(
                "Write a short factual passage (2-4 sentences) that directly "
                "answers the following question. Do not include the question itself."
            )),
            HumanMessage(content=query),
        ]
        try:
            response = await self._llm.ainvoke(messages)
            hypothetical_doc: str = response.content or query
        except Exception as exc:
            logger.warning("HyDE generation failed: %s — falling back to query", exc)
            hypothetical_doc = query

        embedding: list[float] = await asyncio.to_thread(
            self._embedder.embed_query, hypothetical_doc
        )
        meta = RewriteMeta(
            mode="hyde",
            original_query=query,
            rewritten_queries=[hypothetical_doc],
            hypothetical_document=hypothetical_doc,
        )
        return embedding, meta

    # ------------------------------------------------------------------
    # Multi-query
    # ------------------------------------------------------------------

    async def multi_query(self, query: str, n: int = 3) -> tuple[list[str], RewriteMeta]:
        """
        Ask the LLM to rephrase *query* in *n* alternative ways.

        Returns
        -------
        (queries, RewriteMeta)
            queries = [original_query] + alternatives
        """
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

        messages = [
            SystemMessage(content=(
                f"Generate {n} alternative phrasings of the user's question "
                "that preserve the original intent. "
                "Output one phrasing per line, no numbering, no blank lines."
            )),
            HumanMessage(content=query),
        ]
        try:
            response = await self._llm.ainvoke(messages)
            raw: str = response.content or ""
            alternatives = [
                line.strip()
                for line in raw.splitlines()
                if line.strip()
            ][:n]
        except Exception as exc:
            logger.warning("multi_query generation failed: %s — using original only", exc)
            alternatives = []

        all_queries = [query] + alternatives
        meta = RewriteMeta(
            mode="multi_query",
            original_query=query,
            rewritten_queries=all_queries,
        )
        return all_queries, meta

    # ------------------------------------------------------------------
    # Step-back
    # ------------------------------------------------------------------

    async def step_back(self, query: str) -> tuple[list[str], RewriteMeta]:
        """
        Generalise *query* to a broader, higher-level question.

        Returns
        -------
        (queries, RewriteMeta)
            queries = [original_query, step_back_query]
        """
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

        messages = [
            SystemMessage(content=(
                "Rewrite the user's question as a broader, more general question "
                "that captures the underlying concept or domain. "
                "Output only the rewritten question, nothing else."
            )),
            HumanMessage(content=query),
        ]
        try:
            response = await self._llm.ainvoke(messages)
            step_back_query: str = (response.content or query).strip()
        except Exception as exc:
            logger.warning("step_back generation failed: %s — using original only", exc)
            step_back_query = query

        all_queries = [query, step_back_query]
        meta = RewriteMeta(
            mode="step_back",
            original_query=query,
            rewritten_queries=all_queries,
        )
        return all_queries, meta

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    async def rewrite(
        self,
        query: str,
        mode: str,
    ) -> tuple[list[str] | None, list[float] | None, RewriteMeta]:
        """
        Dispatch to the appropriate rewrite strategy.

        Returns
        -------
        (queries_or_None, hyde_embedding_or_None, meta)
            For HyDE: queries=None, hyde_embedding=<vector>
            For multi_query/step_back: queries=<list[str]>, hyde_embedding=None
        """
        if mode == "hyde":
            embedding, meta = await self.hyde(query)
            return None, embedding, meta
        elif mode == "multi_query":
            queries, meta = await self.multi_query(query)
            return queries, None, meta
        elif mode == "step_back":
            queries, meta = await self.step_back(query)
            return queries, None, meta
        else:
            raise ValueError(f"Unknown rewrite mode: {mode!r}")
