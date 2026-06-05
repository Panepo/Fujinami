"""
Self-RAG reflection loop for Fujinami — backed by LangGraph QueryGraph.

``SelfReflector`` is a thin adapter: it constructs a ``QueryGraph`` from the
RAG service's chat LLM and retriever functions, invokes it, and translates
the resulting ``node_trace`` into the ``SelfRagMeta`` structure expected by
``api.py``.

No Semantic Kernel dependencies remain in this file.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from models import SelfRagMeta, SelfRagStep, SourceChunk

_logger = logging.getLogger(__name__)


class SelfReflector:
    """Adaptive RAG query loop backed by ``QueryGraph``."""

    def __init__(self, rag_service: Any, max_iterations: int = 2) -> None:
        self._rag = rag_service
        self._max_iterations = max_iterations
        self._query_graph = self._build_query_graph()

    # ------------------------------------------------------------------
    # QueryGraph construction
    # ------------------------------------------------------------------

    def _build_query_graph(self):
        from graph_engine.query_graph import QueryGraph  # noqa: PLC0415

        async def retriever_fn(question: str, top_k: int):
            raw_rows = await self._rag._raw_vector_results(question, top_k)
            context = "\n\n".join(r.get("text", "") for r in raw_rows)
            sources = []
            for row in raw_rows:
                try:
                    meta = json.loads(row.get("metadata", "{}"))
                except (json.JSONDecodeError, TypeError):
                    meta = {}
                sources.append({
                    "doc_id": row.get("doc_id", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "text": row.get("text", ""),
                })
            return context, sources

        def graph_context_fn(question: str) -> str:
            # _graph_context is a sync method on RagRetriever, safe to call directly
            return self._rag._retriever._graph_context(question)

        return QueryGraph(
            chat_llm=self._rag._chat_service,
            retriever_fn=retriever_fn,
            graph_context_fn=graph_context_fn,
            max_iterations=self._max_iterations,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def query(
        self,
        query: str,
        method: str,
        top_k: int,
    ) -> tuple[str, list[SourceChunk] | None, str | None, SelfRagMeta]:
        """
        Run the QueryGraph and return ``(answer, sources, graphrag_context, meta)``.

        Return type is identical to the previous self-RAG implementation so
        ``api.py`` callers require no changes.
        """
        from graph_engine.state import QueryState  # noqa: PLC0415

        state = QueryState(
            question=query,
            method=method,
            top_k=top_k,
            node_trace=[],
            iterations=0,
        )

        try:
            final: QueryState = await self._query_graph.ainvoke(state)
        except Exception as exc:
            _logger.warning("QueryGraph.ainvoke failed: %s", exc)
            meta = SelfRagMeta(
                needed=True,
                relevant_chunks=0,
                grounded=False,
                iterations=1,
                process_log=[SelfRagStep(
                    step="error",
                    label="QueryGraph error",
                    detail=str(exc),
                    result="Failed",
                    ok=False,
                )],
            )
            return "", None, None, meta

        answer = final.get("answer") or ""
        graphrag_context = final.get("graphrag_context") or None
        raw_sources: list[dict] = final.get("sources") or []
        iterations = final.get("iterations") or 1
        node_trace: list[dict] = final.get("node_trace") or []

        # Convert raw source dicts → SourceChunk
        sources: list[SourceChunk] = [
            SourceChunk(
                doc_id=s.get("doc_id", ""),
                chunk_index=s.get("chunk_index", 0),
                excerpt=s.get("text", "")[:200],
                full_text=s.get("text", ""),
            )
            for s in raw_sources
        ]

        # Translate node_trace → SelfRagStep list
        process_log: list[SelfRagStep] = []
        for entry in node_trace:
            node = entry.get("node", "")
            duration_ms = entry.get("duration_ms", 0)
            detail = entry.get("detail", "")
            process_log.append(SelfRagStep(
                step=node,
                label=node.replace("_", " ").title(),
                detail=detail,
                result=f"{duration_ms} ms",
                ok=True,
            ))

        needs_graph = final.get("needs_graph", False)
        relevant_chunks = len(sources)

        meta = SelfRagMeta(
            needed=True,
            relevant_chunks=relevant_chunks,
            grounded=True,  # QueryGraph grounding is enforced internally
            iterations=iterations,
            process_log=process_log,
        )
        return answer, sources or None, graphrag_context, meta
