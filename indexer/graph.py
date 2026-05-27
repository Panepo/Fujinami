"""Graph extraction logic — moved from indexer.py."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_graph_extraction(
    source_doc: str,
    full_text: str,
    lance_path: Path,
    ollama_url: str,
    extractor_type: str,
    extract_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Extract knowledge graph triples from *full_text* and persist to LanceDB.

    Parameters
    ----------
    source_doc:
        Document filename used as the source identifier in stored triples.
    full_text:
        Full document text to extract triples from.
    lance_path:
        Path to the LanceDB database directory.
    ollama_url:
        Base URL of the Ollama server.
    extractor_type:
        One of ``"spacy"``, ``"llm"``, or ``"hybrid"`` (default).
    extract_model:
        Ollama model name for LLM-based extraction. Empty string means the
        extractor will use its own default.
    chunk_size:
        Token/character chunk size for graph extraction.
    chunk_overlap:
        Token/character overlap between consecutive chunks.
    """
    try:
        from graph_engine.store import LanceDBGraphStore
        from graph_engine.pipeline import GraphPipeline
        from graph_engine.extractors.hybrid_extractor import HybridExtractor
        from graph_engine.extractors.llm_extractor import LLMExtractor
        from graph_engine.extractors.spacy_extractor import SpacyExtractor
    except ImportError as exc:
        logger.warning("graph_engine not available, skipping graph extraction: %s", exc)
        return

    store = LanceDBGraphStore(lance_path)
    extractor_type_lower = extractor_type.lower()

    if extractor_type_lower == "spacy":
        extractor = SpacyExtractor()
        method = "spacy"
    elif extractor_type_lower == "llm":
        extractor = LLMExtractor(
            ollama_url=ollama_url,
            model=extract_model or None,
        )
        method = "llm"
    else:  # hybrid (default)
        extractor = HybridExtractor(
            ollama_url=ollama_url,
            model=extract_model or None,
        )
        method = "hybrid"

    pipeline = GraphPipeline(
        extractor=extractor,
        store=store,
        method=method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    stats = pipeline.run(text=full_text, source_doc=source_doc)
    logger.info(
        "Graph extraction [%s]: chunks=%d raw=%d deduped=%d stored=%d",
        source_doc,
        stats["chunks"],
        stats["raw_triples"],
        stats["deduplicated_triples"],
        stats["stored"],
    )


def remove_graph_triples(lance_path: Path, doc_ids: list[str]) -> None:
    """Delete all graph triples for the given source documents."""
    try:
        from graph_engine.store import LanceDBGraphStore
    except ImportError:
        return
    store = LanceDBGraphStore(lance_path)
    for doc_id in doc_ids:
        removed = store.delete_by_source(doc_id)
        logger.info("Removed %d graph triples for '%s'", removed, doc_id)
