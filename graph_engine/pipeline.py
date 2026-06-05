"""
ExtractionGraph — LangGraph StateGraph wiring chunker → extractor → deduplicator → store.

Used by the indexer to run graph extraction as part of the RAG pipeline.

Usage
-----
    from graph_engine.pipeline import ExtractionGraph, build_pipeline
    from graph_engine.state import ExtractionState
    from graph_engine.store import LanceDBGraphStore
    from graph_engine.extractors.hybrid_extractor import HybridExtractor

    store = LanceDBGraphStore(lance_db_path)
    extractor = HybridExtractor()
    graph = ExtractionGraph(extractor=extractor, store=store)
    result = graph.invoke(ExtractionState(
        raw_text="...", source_doc="my_doc.pdf", method="hybrid"
    ))

    # Factory — builds extractor from method name:
    compiled = build_pipeline(
        method="hybrid", store=store, ollama_url="http://localhost:11434"
    )
    result = compiled.invoke(ExtractionState(raw_text="...", source_doc="my_doc.pdf"))
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from graph_engine.base import BaseExtractor
from graph_engine.chunker import chunk_text
from graph_engine.deduplicator import deduplicate_triples
from graph_engine.models import Triple
from graph_engine.state import ExtractionState
from graph_engine.store import GraphStore

logger = logging.getLogger(__name__)

ExtractorMethod = Literal["spacy", "llm", "hybrid"]


class ExtractionGraph:
    """
    LangGraph-based end-to-end graph extraction pipeline.

    Nodes (linear): chunk_node → extract_node → deduplicate_node → store_node → END

    Parameters
    ----------
    extractor:
        Any ``BaseExtractor`` implementation.
    store:
        A ``GraphStore`` backend (LanceDBGraphStore).
    chunk_size:
        Approximate words per chunk (default 1000).
    chunk_overlap:
        Words repeated between consecutive chunks (default 200).
    method:
        Extraction method label stamped on stored triples.
    """

    def __init__(
        self,
        extractor: BaseExtractor,
        store: GraphStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        method: str = "unknown",
    ) -> None:
        self._extractor = extractor
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._method = method
        self._compiled = self._build()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def invoke(self, state: ExtractionState) -> ExtractionState:
        """Run the full pipeline synchronously and return the final state."""
        return self._compiled.invoke(state)

    def stream(self, state: ExtractionState):
        """Yield per-node output dicts as the graph executes."""
        return self._compiled.stream(state)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build(self):
        chunk_size = self._chunk_size
        chunk_overlap = self._chunk_overlap
        extractor = self._extractor
        store = self._store
        method = self._method

        def chunk_node(state: ExtractionState) -> dict[str, Any]:
            text = state.get("raw_text", "")
            source_doc = state.get("source_doc", "")
            chunks = chunk_text(text, chunk_size, chunk_overlap)
            logger.info("[%s] chunk_node: %d chunks", source_doc, len(chunks))
            return {"chunks": chunks}

        def extract_node(state: ExtractionState) -> dict[str, Any]:
            chunks = state.get("chunks") or []
            source_doc = state.get("source_doc", "")
            raw_triples: list[Triple] = []
            for chunk in chunks:
                try:
                    triples = extractor.extract(chunk, source_doc)
                except Exception as exc:
                    logger.warning("[%s] extractor error: %s", source_doc, exc)
                    triples = []
                raw_triples.extend(triples)
            for t in raw_triples:
                t.method = method
            logger.info("[%s] extract_node: %d raw triples", source_doc, len(raw_triples))
            return {"triples": raw_triples}

        def deduplicate_node(state: ExtractionState) -> dict[str, Any]:
            raw = state.get("triples") or []
            source_doc = state.get("source_doc", "")
            deduped = deduplicate_triples(raw)
            logger.info("[%s] deduplicate_node: %d → %d triples", source_doc, len(raw), len(deduped))
            return {"deduped_triples": deduped}

        def store_node(state: ExtractionState) -> dict[str, Any]:
            deduped = state.get("deduped_triples") or []
            source_doc = state.get("source_doc", "")
            # Remove stale triples for this (source_doc, method) before inserting
            store.delete_by_source_and_method(source_doc, method)
            stored = store.add_triples(deduped)
            logger.info("[%s] store_node: %d triples stored", source_doc, stored)
            return {"stored_count": stored}

        graph = StateGraph(ExtractionState)
        graph.add_node("chunk_node", chunk_node)
        graph.add_node("extract_node", extract_node)
        graph.add_node("deduplicate_node", deduplicate_node)
        graph.add_node("store_node", store_node)

        graph.set_entry_point("chunk_node")
        graph.add_edge("chunk_node", "extract_node")
        graph.add_edge("extract_node", "deduplicate_node")
        graph.add_edge("deduplicate_node", "store_node")
        graph.add_edge("store_node", END)

        return graph.compile()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_pipeline(
    method: str,
    store: GraphStore,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    ollama_url: str | None = None,
    extract_model: str | None = None,
) -> ExtractionGraph:
    """
    Build an :class:`ExtractionGraph` from a method name string.

    Parameters
    ----------
    method:
        One of ``"spacy"``, ``"llm"``, or ``"hybrid"`` (default).
    store:
        GraphStore backend to persist triples into.
    chunk_size / chunk_overlap:
        Chunking parameters forwarded to the graph.
    ollama_url:
        Base URL of the Ollama server (required for llm/hybrid).
    extract_model:
        Ollama model name for LLM-based extraction.

    Returns
    -------
    ExtractionGraph
        Ready-to-use compiled graph (supports ``.invoke()`` and ``.stream()``).
    """
    from graph_engine.extractors.hybrid_extractor import HybridExtractor  # noqa: PLC0415
    from graph_engine.extractors.llm_extractor import LLMExtractor  # noqa: PLC0415
    from graph_engine.extractors.spacy_extractor import SpacyExtractor  # noqa: PLC0415

    method_lower = method.lower()
    if method_lower == "spacy":
        extractor: BaseExtractor = SpacyExtractor()
    elif method_lower == "llm":
        extractor = LLMExtractor(
            ollama_url=ollama_url or "",
            model=extract_model or None,
        )
    else:  # hybrid (default)
        extractor = HybridExtractor(
            ollama_url=ollama_url or "",
            model=extract_model or None,
        )

    return ExtractionGraph(
        extractor=extractor,
        store=store,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        method=method_lower,
    )


# ---------------------------------------------------------------------------
# Legacy shim — keep GraphPipeline importable so any residual callers don't break
# ---------------------------------------------------------------------------

class GraphPipeline:  # noqa: D101
    """Thin backward-compat wrapper around ExtractionGraph."""

    def __init__(
        self,
        extractor: BaseExtractor,
        store: GraphStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        method: str = "unknown",
    ) -> None:
        self._graph = ExtractionGraph(
            extractor=extractor,
            store=store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            method=method,
        )
        self._method = method

    def run(self, text: str, source_doc: str, on_progress=None) -> dict:  # noqa: ANN001
        state = ExtractionState(
            raw_text=text,
            source_doc=source_doc,
            method=self._method,
        )
        final = self._graph.invoke(state)
        chunks_list = final.get("chunks") or []
        raw = final.get("triples") or []
        deduped = final.get("deduped_triples") or []
        stored = final.get("stored_count") or 0
        return {
            "source_doc": source_doc,
            "chunks": len(chunks_list),
            "raw_triples": len(raw),
            "deduplicated_triples": len(deduped),
            "stored": stored,
        }

    def delete_source(self, source_doc: str) -> int:
        """Remove graph data for a document produced by this pipeline's method."""
        return self._graph._store.delete_by_source_and_method(source_doc, self._method)
