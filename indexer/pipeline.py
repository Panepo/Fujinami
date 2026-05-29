"""
RagIndexer — document indexing pipeline as a package.

Orchestrates delta detection, document loading, embedding (via Ollama HTTP),
per-document embedded.json caching, graph extraction, and LanceDB upsert.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import lancedb
from dotenv import load_dotenv

from document_loader import DocumentLoader, SUPPORTED_EXTENSIONS
from indexer.delta import (
    compute_delta,
    load_index_flags,
    load_manifest,
    save_index_flags,
    save_manifest,
)
from indexer.embedder import OllamaEmbedder
from indexer.graph import remove_graph_triples, run_graph_extraction
from indexer.store import (
    _LANCEDB_SCHEMA,
    open_or_create_table,
    remove_from_lancedb,
    upsert_from_embedded_json,
)

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_OLLAMA_INDEX_URL = os.environ["OLLAMA_INDEX_URL"]
_VLM_MODEL = os.environ["VLM_MODEL"]
_VLM_TIMEOUT = float(os.environ.get("VLM_TIMEOUT", "180"))
_EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
_GRAPH_EXTRACTOR = os.environ.get("GRAPH_EXTRACTOR", "hybrid")
_EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL", os.environ.get("INDEX_MODEL", ""))
_GRAPH_CHUNK_SIZE = int(os.environ.get("GRAPH_CHUNK_SIZE", "400"))
_GRAPH_CHUNK_OVERLAP = int(os.environ.get("GRAPH_CHUNK_OVERLAP", "80"))


class RagIndexer:
    """
    Handles incremental document indexing into LanceDB + graph triple extraction.

    Parameters
    ----------
    collection_name:
        Name of the document collection.
    root_dir:
        Root directory for data and ragdata storage.
    lance_db_path:
        Path to the LanceDB database directory.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        root_dir: str | Path | None = None,
        lance_db_path: str | Path | None = None,
    ) -> None:
        self._root_dir = Path(root_dir) if root_dir else Path(__file__).parent.parent
        self._collection_name = collection_name

        if collection_name is not None:
            self._ragdata_dir = self._root_dir / "ragdata" / collection_name
            self._data_dir = self._root_dir / "data" / collection_name
        else:
            self._ragdata_dir = self._root_dir / "ragdata"
            self._data_dir = self._root_dir / "data"

        lance_path = (
            Path(lance_db_path) if lance_db_path else self._ragdata_dir / "lancedb"
        )
        self._lance_path = lance_path
        self._manifest_path = lance_path / "file_manifest.json"

        # --- OllamaEmbedder setup (no SK dependency) ---
        self._embedder = OllamaEmbedder(
            model=_EMBEDDING_MODEL,
            ollama_base_url=_OLLAMA_INDEX_URL,
        )

        # --- LanceDB setup ---
        lance_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(lance_path))
        self._table = open_or_create_table(self._db, _LANCEDB_SCHEMA)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def index_documents(
        self,
        documents_dir: str | Path | None = None,
        entity_types: list[str] | None = None,
        mode: str = "all",
        force: bool = False,
    ) -> None:
        """
        Incremental indexing pipeline:

        1. Compute per-file delta using SHA-256 content hash.
        2. If nothing changed, return immediately.
        3. Delete LanceDB rows and graph triples for removed/modified sources.
        4. Load and convert only new/modified files via DocumentLoader.
        5. Run graph extraction on full document text.
        6. Embed chunks via OllamaEmbedder, write per-document embedded.json.
        7. Upsert from embedded.json into LanceDB.
        8. Save updated manifest.

        Parameters
        ----------
        mode:
            ``"all"``    — run both vector (LanceDB) and graph extraction.
            ``"vector"`` — run only LanceDB embedding; skip graph extraction.
            ``"graph"``  — run only graph extraction; skip LanceDB embedding.
        force:
            When ``True``, ignore the manifest and reprocess all files from
            scratch.
        """
        run_vector = mode in ("vector", "all")
        run_graph = mode in ("graph", "all")

        if documents_dir is None:
            documents_dir = self._data_dir
        documents_dir = Path(documents_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        if force:
            changed_sources: set[str] = {
                fp.name
                for fp in documents_dir.rglob("*")
                if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTENSIONS
            }
            if not changed_sources:
                logger.info("Force reindex: no supported files found in %s", documents_dir)
                return
            if run_vector:
                remove_from_lancedb(self._table, list(changed_sources))
            if run_graph:
                remove_graph_triples(self._lance_path, list(changed_sources))
            logger.info("Force reindex: processing %d file(s)", len(changed_sources))
        else:
            stored_manifest = load_manifest(self._manifest_path)
            new_files, modified_files, deleted_files, _ = compute_delta(
                documents_dir, stored_manifest
            )

            if not new_files and not modified_files and not deleted_files:
                logger.info("No changes detected, skipping indexing")
                return

            logger.info(
                "Delta — new: %d, modified: %d, deleted: %d",
                len(new_files),
                len(modified_files),
                len(deleted_files),
            )

            removed_sources = deleted_files | modified_files
            changed_sources = new_files | modified_files

            if removed_sources:
                if run_vector:
                    remove_from_lancedb(self._table, list(removed_sources))
                if run_graph:
                    remove_graph_triples(self._lance_path, list(removed_sources))

            if not changed_sources:
                # Only deletions
                save_manifest(documents_dir, self._manifest_path)
                save_index_flags(
                    self._ragdata_dir,
                    vector_indexed=True if run_vector else None,
                    graph_indexed=True if run_graph else None,
                )
                return

        # Step 4 — load only changed files
        loader = DocumentLoader(
            ollama_base_url=_OLLAMA_INDEX_URL,
            vlm_model=_VLM_MODEL,
            request_timeout=_VLM_TIMEOUT,
        )
        doc_chunks = await asyncio.to_thread(
            loader.load_directory, documents_dir, files_filter=changed_sources
        )

        failed_sources = changed_sources - set(doc_chunks.keys())
        if failed_sources:
            logger.warning(
                "Failed to load %d file(s), will retry on next run: %s",
                len(failed_sources),
                failed_sources,
            )

        if not doc_chunks:
            logger.warning("No content loaded for changed sources")
            save_manifest(documents_dir, self._manifest_path, exclude=changed_sources)
            return

        # Step 5 — graph extraction
        if run_graph:
            for filename, chunks in doc_chunks.items():
                full_text = "\n\n".join(
                    c.get("chunk_text_original", "")
                    for c in chunks
                    if c.get("chunk_text_original")
                )
                if full_text.strip():
                    await asyncio.to_thread(
                        run_graph_extraction,
                        filename,
                        full_text,
                        self._lance_path,
                        _OLLAMA_INDEX_URL,
                        _GRAPH_EXTRACTOR,
                        _EXTRACT_MODEL,
                        _GRAPH_CHUNK_SIZE,
                        _GRAPH_CHUNK_OVERLAP,
                    )

        # Step 6 & 7 — embed, save embedded.json, upsert to LanceDB
        if run_vector:
            for filename, chunks in doc_chunks.items():
                if not chunks:
                    continue
                embedded_path = await asyncio.to_thread(
                    self._embed_and_save, filename, chunks
                )
                self._table = upsert_from_embedded_json(
                    self._db, self._table, embedded_path
                )

        # Step 8 — persist manifest
        save_manifest(documents_dir, self._manifest_path, exclude=failed_sources)
        save_index_flags(
            self._ragdata_dir,
            vector_indexed=True if run_vector else None,
            graph_indexed=True if run_graph else None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_and_save(self, filename: str, chunks: list[dict]) -> Path:
        """Embed *chunks* and write ``embedded.json`` atomically.

        Uses the cached file if it already exists and all ``chunk_hash`` values
        match the incoming chunks (i.e., the document content is unchanged).

        Returns
        -------
        Path
            Path to the written (or cached) embedded.json file.
        """
        doc_stem = Path(filename).stem
        embedded_dir = self._ragdata_dir / "embedded"
        embedded_dir.mkdir(parents=True, exist_ok=True)
        output_path = embedded_dir / f"{doc_stem}.embedded.json"

        # Cache check: all chunk_hashes present and identical → skip re-embedding
        if output_path.exists():
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
                existing_hashes = {
                    c.get("chunk_hash")
                    for c in existing.get("chunks", [])
                    if c.get("chunk_hash")
                }
                current_hashes = {
                    c.get("chunk_hash")
                    for c in chunks
                    if c.get("chunk_hash")
                }
                if current_hashes and current_hashes == existing_hashes:
                    logger.info(
                        "Cache hit for '%s', skipping re-embedding (%d chunks)",
                        filename,
                        len(chunks),
                    )
                    return output_path
            except (json.JSONDecodeError, OSError, KeyError):
                pass  # Invalid cache — fall through to re-embed

        # Embed
        texts = [
            c.get("chunk_text_embedded") or c.get("chunk_text_original", "")
            for c in chunks
        ]
        embeddings = self._embedder.embed(texts)

        # Build embedded.json payload
        embedded_chunks = []
        for chunk, emb in zip(chunks, embeddings):
            embedded_chunk = dict(chunk)
            embedded_chunk["embedding"] = emb.tolist()
            embedded_chunks.append(embedded_chunk)

        payload = {
            "model": self._embedder.model_name,
            "dimension": self._embedder.dimension,
            "device": "ollama",
            "doc_stem": doc_stem,
            "filename": filename,
            "chunks": embedded_chunks,
        }

        # Atomic write: temp file in same directory → rename
        tmp_fd, tmp_path = tempfile.mkstemp(dir=embedded_dir, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            Path(tmp_path).replace(output_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info(
            "Saved embedded.json for '%s' with %d chunks",
            filename,
            len(embedded_chunks),
        )
        return output_path

    async def rebuild_from_embedded(self) -> int:
        """Rebuild the LanceDB table from all cached embedded.json files.

        Drops all existing rows, then upserts every
        ``ragdata/{collection}/embedded/*.embedded.json`` found on disk.
        Does **not** re-embed or re-parse documents — uses the cached
        ``chunk_text_embedded`` text that was stored during the original
        indexing run.

        Returns
        -------
        int
            Number of embedded.json files processed.
        """
        embedded_dir = self._ragdata_dir / "embedded"
        paths = sorted(embedded_dir.glob("*.embedded.json")) if embedded_dir.exists() else []
        if not paths:
            logger.warning("rebuild_from_embedded: no embedded.json files found in %s", embedded_dir)
            return 0

        # Drop and recreate the table so stale rows are gone
        import lancedb as _lancedb  # noqa: PLC0415
        from indexer.store import _TABLE_NAME as _TN  # noqa: PLC0415

        if _TN in self._db.table_names():
            self._db.drop_table(_TN)
            logger.info("rebuild_from_embedded: dropped existing LanceDB table '%s'", _TN)
        self._table = None

        for path in paths:
            self._table = upsert_from_embedded_json(self._db, self._table, path)
            logger.info("rebuild_from_embedded: upserted %s", path.name)

        logger.info("rebuild_from_embedded: done, processed %d file(s)", len(paths))
        return len(paths)
