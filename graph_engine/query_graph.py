"""
QueryGraph — LangGraph StateGraph for adaptive RAG query flow.

Architecture (mirrors reference vector_retrieve → evaluate_context → [conditional] → generate_answer):

    START
      │
      ▼
  vector_retrieve_node  ─── fills context + sources
      │
      ▼
  evaluate_context_node ─── LLM YES/NO: does context suffice?
      │
      ├─ needs_graph=True  ──► graph_retrieve_node ──► generate_answer_node
      │
      └─ needs_graph=False ──────────────────────────► generate_answer_node
                                                              │
                                                             END

For method="graph": START → graph_retrieve_node → generate_answer_node → END
(bypasses vector_retrieve and evaluate_context).

Each node appends a timing entry to state.node_trace so callers can surface
real-time progress via SSE or build a UI flow trace.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from graph_engine.state import QueryState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_EVALUATE_PROMPT = (
    "You are evaluating whether the retrieved context is sufficient to answer the question.\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Is the context sufficient to answer the question without additional knowledge-graph data? "
    "Reply with ONLY 'YES' or 'NO'."
)

_GENERATE_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the provided context. "
    "If the context does not contain enough information, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


# ---------------------------------------------------------------------------
# QueryGraph
# ---------------------------------------------------------------------------

class QueryGraph:
    """
    LangGraph-based adaptive RAG query pipeline.

    Parameters
    ----------
    chat_llm:
        A ``ChatOllama`` instance (langchain-ollama).
    retriever_fn:
        Async callable ``(question: str, top_k: int) -> (context_str: str, sources: list[dict])``.
        Called by ``vector_retrieve_node``.
    graph_context_fn:
        Sync callable ``(question: str) -> str``.
        Called by ``graph_retrieve_node``.
    max_iterations:
        Maximum query iterations for the SelfReflector loop (default 2).
        The QueryGraph itself runs once; the caller controls re-invocation.
    """

    def __init__(
        self,
        chat_llm: Any,
        retriever_fn: Callable,
        graph_context_fn: Callable,
        max_iterations: int = 2,
    ) -> None:
        self._chat_llm = chat_llm
        self._retriever_fn = retriever_fn
        self._graph_context_fn = graph_context_fn
        self._max_iterations = max_iterations
        self._compiled = self._build()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def ainvoke(self, state: QueryState) -> QueryState:
        """Run the query graph asynchronously and return the final state."""
        return await self._compiled.ainvoke(state)

    def invoke(self, state: QueryState) -> QueryState:
        """Run the query graph synchronously (blocks the event loop — prefer ainvoke)."""
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(state))

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trace_start(node_name: str) -> float:
        return time.time()

    @staticmethod
    def _trace_entry(node_name: str, started_at: float, detail: str = "") -> dict:
        duration_ms = int((time.time() - started_at) * 1000)
        return {
            "node": node_name,
            "started_at": started_at,
            "duration_ms": duration_ms,
            "detail": detail,
        }

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build(self):
        chat_llm = self._chat_llm
        retriever_fn = self._retriever_fn
        graph_context_fn = self._graph_context_fn

        async def vector_retrieve_node(state: QueryState) -> dict[str, Any]:
            started = time.time()
            question = state.get("question", "")
            top_k = state.get("top_k", 5)
            trace = list(state.get("node_trace") or [])

            context, sources = await retriever_fn(question, top_k)
            trace.append(_trace_entry("vector_retrieve", started, f"{len(sources)} chunks retrieved"))
            return {"context": context, "sources": sources, "node_trace": trace}

        async def evaluate_context_node(state: QueryState) -> dict[str, Any]:
            started = time.time()
            context = state.get("context", "")
            question = state.get("question", "")
            trace = list(state.get("node_trace") or [])
            needs_graph = False

            if context.strip():
                try:
                    from langchain_core.messages import HumanMessage  # noqa: PLC0415
                    prompt = _EVALUATE_PROMPT.format(context=context[:2000], question=question)
                    response = await chat_llm.ainvoke([HumanMessage(content=prompt)])
                    reply = (response.content or "").strip().upper()
                    needs_graph = reply.startswith("NO")
                except Exception as exc:
                    logger.debug("evaluate_context_node LLM error: %s", exc)
                    needs_graph = False
            else:
                needs_graph = True  # no context → definitely need graph

            detail = "Graph needed" if needs_graph else "Context sufficient"
            trace.append(_trace_entry("evaluate_context", started, detail))
            return {"needs_graph": needs_graph, "node_trace": trace}

        async def graph_retrieve_node(state: QueryState) -> dict[str, Any]:
            started = time.time()
            question = state.get("question", "")
            trace = list(state.get("node_trace") or [])

            try:
                graphrag_context = await asyncio.to_thread(graph_context_fn, question)
            except Exception as exc:
                logger.debug("graph_retrieve_node error: %s", exc)
                graphrag_context = ""

            trace.append(_trace_entry("graph_retrieve", started, f"{len(graphrag_context)} chars of graph context"))
            return {"graphrag_context": graphrag_context, "node_trace": trace}

        async def generate_answer_node(state: QueryState) -> dict[str, Any]:
            started = time.time()
            question = state.get("question", "")
            context = state.get("context", "")
            graphrag_context = state.get("graphrag_context", "")
            iterations = state.get("iterations", 0) + 1
            trace = list(state.get("node_trace") or [])

            # Merge context sources
            parts: list[str] = []
            if context:
                parts.append(context)
            if graphrag_context:
                parts.append(f"Graph context:\n{graphrag_context}")
            merged = "\n\n".join(parts) or "No context available."

            try:
                from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415
                messages = [
                    SystemMessage(content=(
                        "You are a helpful assistant. Answer the user's question using only "
                        "the provided context. If the context does not contain enough information, say so."
                    )),
                    HumanMessage(content=f"Context:\n{merged}\n\nQuestion: {question}"),
                ]
                response = await chat_llm.ainvoke(messages)
                answer = response.content or ""
            except Exception as exc:
                logger.warning("generate_answer_node LLM error: %s", exc)
                answer = ""

            trace.append(_trace_entry("generate_answer", started, f"{len(answer)} chars"))
            return {"answer": answer, "iterations": iterations, "node_trace": trace}

        # Routing
        def route_after_evaluate(state: QueryState) -> str:
            return "graph_retrieve_node" if state.get("needs_graph") else "generate_answer_node"

        def route_entry(state: QueryState) -> str:
            method = (state.get("method") or "").lower()
            return "graph_retrieve_node" if method == "graph" else "vector_retrieve_node"

        # Helpers used inside nodes (need to be in scope)
        def _trace_entry(node_name: str, started_at: float, detail: str = "") -> dict:
            duration_ms = int((time.time() - started_at) * 1000)
            return {
                "node": node_name,
                "started_at": started_at,
                "duration_ms": duration_ms,
                "detail": detail,
            }

        graph = StateGraph(QueryState)
        graph.add_node("vector_retrieve_node", vector_retrieve_node)
        graph.add_node("evaluate_context_node", evaluate_context_node)
        graph.add_node("graph_retrieve_node", graph_retrieve_node)
        graph.add_node("generate_answer_node", generate_answer_node)

        # Entry routing based on method
        graph.set_conditional_entry_point(
            route_entry,
            {
                "vector_retrieve_node": "vector_retrieve_node",
                "graph_retrieve_node": "graph_retrieve_node",
            },
        )

        graph.add_edge("vector_retrieve_node", "evaluate_context_node")
        graph.add_conditional_edges(
            "evaluate_context_node",
            route_after_evaluate,
            {
                "graph_retrieve_node": "graph_retrieve_node",
                "generate_answer_node": "generate_answer_node",
            },
        )
        graph.add_edge("graph_retrieve_node", "generate_answer_node")
        graph.add_edge("generate_answer_node", END)

        return graph.compile()
