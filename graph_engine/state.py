"""
LangGraph state definitions for graph_engine.

ExtractionState — for the indexing/extraction pipeline.
QueryState      — for the adaptive query pipeline (Self-RAG / LangGraph flow).
"""
from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class ExtractionState(TypedDict, total=False):
    """State threaded through the ExtractionGraph pipeline."""

    raw_text: str
    source_doc: str
    method: str
    chunk_size: int
    chunk_overlap: int
    # populated by nodes
    chunks: list[str]
    triples: list[Any]           # list[Triple]
    deduped_triples: list[Any]   # list[Triple]
    stored_count: int
    error: str | None


class QueryState(TypedDict, total=False):
    """State threaded through the QueryGraph pipeline."""

    question: str
    method: str
    top_k: int
    context: str
    sources: list[dict]
    graphrag_context: str
    needs_graph: bool
    answer: str
    iterations: int
    node_trace: list[dict]  # [{node, started_at, duration_ms, detail}]
