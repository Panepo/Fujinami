"""
GraphPipeline — wires chunker → extractor → deduplicator → store.

Used by ragService to run graph extraction as a feature alongside
(not replacing) the existing GraphRAG CLI indexing.

Usage
-----
    from graph_engine.pipeline import GraphPipeline
    from graph_engine.extractors.hybrid_extractor import HybridExtractor
    from graph_engine.store import LanceDBGraphStore

    store = LanceDBGraphStore(lance_db_path)
    pipeline = GraphPipeline(
        extractor=HybridExtractor(),
        store=store,
    )
    stats = pipeline.run(text="...", source_doc="my_doc.pdf")
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal

from graph_engine.base import BaseExtractor
from graph_engine.chunker import chunk_text
from graph_engine.deduplicator import deduplicate_triples
from graph_engine.models import Triple
from graph_engine.store import GraphStore

logger = logging.getLogger(__name__)

ExtractorMethod = Literal["spacy", "llm", "hybrid"]


class GraphPipeline:
    """
    End-to-end graph extraction pipeline.

    Parameters
    ----------
    extractor:
        Any ``BaseExtractor`` implementation (SpacyExtractor,
        LLMExtractor, or HybridExtractor).
    store:
        A ``GraphStore`` backend (LanceDBGraphStore or AGEGraphStore).
    chunk_size:
        Approximate words per chunk.
    chunk_overlap:
        Words repeated between consecutive chunks.
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

    def run(
        self,
        text: str,
        source_doc: str,
        on_progress: Callable[[int | None, str], None] | None = None,
    ) -> dict:
        """
        Run the full pipeline on a single document's text.

        Parameters
        ----------
        text:
            Plain-text content of the document (all pages merged).
        source_doc:
            Document identifier — stored on every node and edge.
        on_progress:
            Optional callback ``(percent: int|None, stage: str) -> None`` called
            at key milestones so callers can track progress.
            Pass ``None`` for percent to signal indeterminate progress.

        Returns
        -------
        dict with keys: chunks, raw_triples, deduplicated_triples, stored
        """
        def _report(pct: int | None, stage: str) -> None:
            if on_progress:
                on_progress(pct, stage)

        # 1. Chunk
        _report(5, f"Chunking — {source_doc}")
        chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
        logger.info("[%s] %d chunks created", source_doc, len(chunks))

        # 2. Extract triples from every chunk — report per-chunk progress (5 → 80%)
        raw_triples: list[Triple] = []
        total = max(len(chunks), 1)
        for i, chunk in enumerate(chunks):
            chunk_start_pct = 5 + int((i / total) * 75)
            chunk_end_pct = 5 + int(((i + 1) / total) * 75)

            def _on_batch(batch_num: int, batch_total: int, _cs=chunk_start_pct, _ce=chunk_end_pct) -> None:
                batch_pct = _cs + int((batch_num / batch_total) * (_ce - _cs))
                _report(batch_pct, f"Extracting — chunk {i + 1}/{total}, batch {batch_num}/{batch_total} ({source_doc})")

            if total == 1:
                _report(None, f"Extracting — {source_doc} (waiting for model…)")
            else:
                _report(chunk_start_pct, f"Extracting — chunk {i + 1}/{total} ({source_doc})")

            # Pass on_batch for extractors that support it (HybridExtractor)
            try:
                chunk_triples = self._extractor.extract(chunk, source_doc, on_batch=_on_batch)
            except TypeError:
                chunk_triples = self._extractor.extract(chunk, source_doc)

            logger.debug("[%s] chunk %d → %d triples", source_doc, i, len(chunk_triples))
            raw_triples.extend(chunk_triples)
            _report(chunk_end_pct, f"Extracted chunk {i + 1}/{total} — {len(chunk_triples)} triples ({source_doc})")

        logger.info("[%s] %d raw triples extracted", source_doc, len(raw_triples))

        # 2b. Stamp extraction method on all triples
        for t in raw_triples:
            t.method = self._method

        # 3. Deduplicate
        _report(85, f"Deduplicating triples — {source_doc}")
        deduped = deduplicate_triples(raw_triples)
        logger.info("[%s] %d triples after deduplication", source_doc, len(deduped))

        # 4. Store — clear old triples for this (source_doc, method) first so
        #    re-indexing fully replaces stale data instead of accumulating duplicates.
        _report(95, f"Storing to LanceDB — {source_doc}")
        self._store.delete_by_source_and_method(source_doc, self._method)
        stored = self._store.add_triples(deduped)
        logger.info("[%s] %d triples stored", source_doc, stored)

        _report(100, f"Done — {source_doc}")
        return {
            "source_doc": source_doc,
            "chunks": len(chunks),
            "raw_triples": len(raw_triples),
            "deduplicated_triples": len(deduped),
            "stored": stored,
        }

    def delete_source(self, source_doc: str) -> int:
        """Remove graph data for a document produced by this pipeline's method."""
        return self._store.delete_by_source_and_method(source_doc, self._method)


# ---------------------------------------------------------------------------
# Factory helper — build a pipeline by method name
# ---------------------------------------------------------------------------


def build_pipeline(
    method: ExtractorMethod,
    store: GraphStore,
    spacy_model: str = "en_core_web_sm",
    ollama_url: str | None = None,
    llm_model: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> GraphPipeline:
    """
    Convenience factory to create a ``GraphPipeline`` by method name.

    Parameters
    ----------
    method:
        ``"spacy"`` | ``"llm"`` | ``"hybrid"``
    store:
        A ``GraphStore`` instance.
    spacy_model, ollama_url, llm_model:
        Passed to the extractor.
    """
    if method == "spacy":
        from graph_engine.extractors.spacy_extractor import SpacyExtractor  # noqa: PLC0415

        extractor: BaseExtractor = SpacyExtractor(model_name=spacy_model)

    elif method == "llm":
        from graph_engine.extractors.llm_extractor import LLMExtractor  # noqa: PLC0415

        extractor = LLMExtractor(ollama_url=ollama_url, model=llm_model)

    else:  # hybrid (default)
        from graph_engine.extractors.hybrid_extractor import HybridExtractor  # noqa: PLC0415

        extractor = HybridExtractor(
            spacy_model=spacy_model,
            ollama_url=ollama_url,
            model=llm_model,
        )

    return GraphPipeline(
        extractor=extractor,
        store=store,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        method=method,
    )
