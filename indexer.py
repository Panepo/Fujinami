"""
RagIndexer — document indexing: delta detection, loading, graph extraction, LanceDB upsert.

Replaces the indexing logic from the monolithic RagService.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import pyarrow as pa
from dotenv import load_dotenv

from document_loader import DocumentLoader, SUPPORTED_EXTENSIONS

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LanceDB schema for document vectors
# ---------------------------------------------------------------------------

_LANCEDB_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 768)),
        pa.field("metadata", pa.string()),
    ]
)

_TABLE_NAME = "documents"

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
        self._root_dir = Path(root_dir) if root_dir else Path(__file__).parent
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

        # --- Semantic Kernel embedding setup ---
        from semantic_kernel import Kernel
        from semantic_kernel.connectors.ai.ollama import OllamaTextEmbedding

        self._kernel = Kernel()
        self._embedding_service = OllamaTextEmbedding(
            ai_model_id=_EMBEDDING_MODEL,
            host=_OLLAMA_INDEX_URL,
            service_id="embedding",
        )
        self._kernel.add_service(self._embedding_service)

        # --- LanceDB setup ---
        import lancedb  # noqa: PLC0415

        lance_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(lance_path))

        if _TABLE_NAME in self._db.table_names():
            existing = self._db.open_table(_TABLE_NAME)
            existing_vector_type = existing.schema.field("vector").type
            expected_vector_type = _LANCEDB_SCHEMA.field("vector").type
            if existing_vector_type != expected_vector_type:
                logger.warning(
                    "LanceDB table '%s' has schema mismatch (vector %s vs expected %s). "
                    "Dropping and recreating.",
                    _TABLE_NAME,
                    existing_vector_type,
                    expected_vector_type,
                )
                self._db.drop_table(_TABLE_NAME)
                self._table = None
                logger.info("LanceDB table '%s' will be recreated on first index", _TABLE_NAME)
            else:
                self._table = existing
                logger.info("Opened existing LanceDB table '%s'", _TABLE_NAME)
        else:
            self._table = None
            logger.info("LanceDB table '%s' will be created on first index", _TABLE_NAME)

        self._manifest_path = lance_path / "file_manifest.json"

    # ------------------------------------------------------------------
    # Index-flag helpers
    # ------------------------------------------------------------------

    def _load_index_flags(self) -> dict:
        flags_path = self._ragdata_dir / "index_flags.json"
        if not flags_path.exists():
            return {"vector_indexed": False, "graph_indexed": False}
        try:
            return json.loads(flags_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"vector_indexed": False, "graph_indexed": False}

    def _save_index_flags(
        self,
        *,
        vector_indexed: Optional[bool] = None,
        graph_indexed: Optional[bool] = None,
    ) -> None:
        flags = self._load_index_flags()
        if vector_indexed is not None:
            flags["vector_indexed"] = vector_indexed
        if graph_indexed is not None:
            flags["graph_indexed"] = graph_indexed
        self._ragdata_dir.mkdir(parents=True, exist_ok=True)
        (self._ragdata_dir / "index_flags.json").write_text(
            json.dumps(flags, indent=2), encoding="utf-8"
        )
        logger.info("Index flags saved: %s", flags)

    # ------------------------------------------------------------------
    # Indexing
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
        4. Load and convert only new/modified files via DocumentLoader (5-stage pipeline).
        5. Run graph extraction (graph_engine) on full document text.
        6. Embed chunk_text_embedded and add vectors into LanceDB.
        7. Save updated manifest.

        Parameters
        ----------
        mode:
            ``"all"``    — run both vector (LanceDB) and graph extraction.
            ``"vector"`` — run only LanceDB embedding; skip graph extraction.
            ``"graph"``  — run only graph extraction; skip LanceDB embedding.
        force:
            When ``True``, ignore the manifest and reprocess all files from
            scratch.  Existing LanceDB rows and graph triples for every file
            in *documents_dir* are deleted before re-indexing.
        """
        run_vector = mode in ("vector", "all")
        run_graph = mode in ("graph", "all")
        if documents_dir is None:
            documents_dir = self._data_dir
        documents_dir = Path(documents_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        if force:
            # Force full reindex: treat every on-disk file as changed.
            changed_sources: set[str] = {
                fp.name
                for fp in documents_dir.rglob("*")
                if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTENSIONS
            }
            if not changed_sources:
                logger.info("Force reindex: no supported files found in %s", documents_dir)
                return
            # Remove all existing rows so we don't accumulate duplicates.
            if run_vector:
                self._remove_from_lancedb(list(changed_sources))
            if run_graph:
                self._remove_graph_triples(list(changed_sources))
            logger.info("Force reindex: processing %d file(s)", len(changed_sources))
        else:
            # Step 1 — delta detection (SHA-256 hash)
            stored_manifest = self._load_manifest()
            new_files, modified_files, deleted_files, _ = self._compute_delta(
                documents_dir, stored_manifest
            )

            if not new_files and not modified_files and not deleted_files:
                logger.info("No changes detected, skipping indexing")
                return

            logger.info(
                "Delta — new: %d, modified: %d, deleted: %d",
                len(new_files), len(modified_files), len(deleted_files),
            )

            removed_sources = deleted_files | modified_files
            changed_sources = new_files | modified_files

            # Step 3 — remove stale LanceDB rows and graph triples
            if removed_sources:
                if run_vector:
                    self._remove_from_lancedb(list(removed_sources))
                if run_graph:
                    self._remove_graph_triples(list(removed_sources))

            if not changed_sources:
                # Only deletions
                self._save_manifest(documents_dir)
                self._save_index_flags(
                    vector_indexed=True if run_vector else None,
                    graph_indexed=True if run_graph else None,
                )
                return

        # Step 4 — load only changed files (run in thread to avoid blocking event loop)
        loader = DocumentLoader(
            ollama_base_url=_OLLAMA_INDEX_URL,
            vlm_model=_VLM_MODEL,
            request_timeout=_VLM_TIMEOUT,
        )
        doc_chunks = await asyncio.to_thread(
            loader.load_directory, documents_dir, files_filter=changed_sources
        )

        # Identify files that failed to load — exclude from manifest so they
        # are retried on the next indexing run instead of being silently skipped.
        failed_sources = changed_sources - set(doc_chunks.keys())
        if failed_sources:
            logger.warning(
                "Failed to load %d file(s), will retry on next run: %s",
                len(failed_sources), failed_sources,
            )

        if not doc_chunks:
            logger.warning("No content loaded for changed sources")
            # Exclude failed files from the manifest so they are not silently skipped.
            self._save_manifest(documents_dir, exclude=changed_sources)
            return

        # Step 5 — graph extraction
        if run_graph:
            for filename, chunks in doc_chunks.items():
                full_text = "\n\n".join(
                    c.get("chunk_text_original", "") for c in chunks
                    if c.get("chunk_text_original")
                )
                if full_text.strip():
                    await asyncio.to_thread(
                        self._run_graph_extraction, filename, full_text
                    )

        # Step 6 — add new chunks to LanceDB
        if run_vector:
            await self._upsert_to_lancedb(doc_chunks)

        # Step 7 — persist manifest (excluding files that failed to load)
        self._save_manifest(documents_dir, exclude=failed_sources)

        self._save_index_flags(
            vector_indexed=True if run_vector else None,
            graph_indexed=True if run_graph else None,
        )

    def _run_graph_extraction(self, source_doc: str, full_text: str) -> None:
        """Extract knowledge graph triples from *full_text* and persist to LanceDB."""
        try:
            from graph_engine.store import LanceDBGraphStore
            from graph_engine.pipeline import GraphPipeline
            from graph_engine.extractors.hybrid_extractor import HybridExtractor
            from graph_engine.extractors.llm_extractor import LLMExtractor
            from graph_engine.extractors.spacy_extractor import SpacyExtractor
        except ImportError as exc:
            logger.warning("graph_engine not available, skipping graph extraction: %s", exc)
            return

        store = LanceDBGraphStore(self._lance_path)
        extractor_type = _GRAPH_EXTRACTOR.lower()

        if extractor_type == "spacy":
            extractor = SpacyExtractor()
            method = "spacy"
        elif extractor_type == "llm":
            extractor = LLMExtractor(
                ollama_url=_OLLAMA_INDEX_URL,
                model=_EXTRACT_MODEL or None,
            )
            method = "llm"
        else:  # hybrid (default)
            extractor = HybridExtractor(
                ollama_url=_OLLAMA_INDEX_URL,
                model=_EXTRACT_MODEL or None,
            )
            method = "hybrid"

        pipeline = GraphPipeline(
            extractor=extractor,
            store=store,
            method=method,
            chunk_size=_GRAPH_CHUNK_SIZE,
            chunk_overlap=_GRAPH_CHUNK_OVERLAP,
        )
        stats = pipeline.run(text=full_text, source_doc=source_doc)
        logger.info(
            "Graph extraction [%s]: chunks=%d raw=%d deduped=%d stored=%d",
            source_doc, stats["chunks"], stats["raw_triples"],
            stats["deduplicated_triples"], stats["stored"],
        )

    def _remove_graph_triples(self, doc_ids: list[str]) -> None:
        """Delete all graph triples for the given source documents."""
        try:
            from graph_engine.store import LanceDBGraphStore
        except ImportError:
            return
        store = LanceDBGraphStore(self._lance_path)
        for doc_id in doc_ids:
            removed = store.delete_by_source(doc_id)
            logger.info("Removed %d graph triples for '%s'", removed, doc_id)

    # ------------------------------------------------------------------
    # Manifest helpers (SHA-256 content hash)
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, str]:
        """Load ``file_manifest.json`` → ``{filename: sha256_hex}``."""
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read manifest, treating as empty: %s", exc)
            return {}

    def _compute_delta(
        self,
        documents_dir: Path,
        stored_manifest: dict[str, str],
    ) -> tuple[set[str], set[str], set[str], set[str]]:
        """Compare on-disk files against *stored_manifest* using SHA-256 content hashes."""
        on_disk: dict[str, str] = {}
        for file_path in documents_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
                on_disk[file_path.name] = file_hash

        on_disk_names = set(on_disk)
        stored_names = set(stored_manifest)

        new_files = on_disk_names - stored_names
        deleted_files = stored_names - on_disk_names
        modified_files: set[str] = set()
        unchanged_files: set[str] = set()

        for name in on_disk_names & stored_names:
            if on_disk[name] != stored_manifest[name]:
                modified_files.add(name)
            else:
                unchanged_files.add(name)

        return new_files, modified_files, deleted_files, unchanged_files

    def _save_manifest(
        self,
        documents_dir: Path,
        exclude: set[str] | None = None,
    ) -> None:
        """Write fresh ``file_manifest.json`` with SHA-256 hashes for all on-disk files.

        Parameters
        ----------
        exclude:
            Filenames to omit from the manifest (e.g. files that failed to
            load).  Omitted files will appear as *new* on the next run and
            will be retried.
        """
        exclude = exclude or set()
        manifest: dict[str, str] = {}
        for file_path in documents_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                if file_path.name not in exclude:
                    manifest[file_path.name] = hashlib.sha256(file_path.read_bytes()).hexdigest()
        self._manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Manifest saved with %d entries", len(manifest))

    # ------------------------------------------------------------------
    # LanceDB helpers
    # ------------------------------------------------------------------

    def _remove_from_lancedb(self, doc_ids: list[str]) -> None:
        """Delete all LanceDB rows whose ``doc_id`` is in *doc_ids*."""
        if self._table is None:
            return
        for doc_id in doc_ids:
            try:
                safe_id = doc_id.replace("'", "\\'")
                self._table.delete(f"doc_id = '{safe_id}'")
                logger.info("Removed LanceDB rows for doc_id '%s'", doc_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Delete failed for doc_id '%s': %s", doc_id, exc)

    async def _upsert_to_lancedb(
        self, doc_chunks: dict[str, list[dict]]
    ) -> None:
        """Embed ``chunk_text_embedded`` and upsert rows into LanceDB."""
        rows: list[dict] = []

        for filename, chunks in doc_chunks.items():
            if not chunks:
                continue

            texts_to_embed = [
                c.get("chunk_text_embedded") or c.get("chunk_text_original", "")
                for c in chunks
            ]
            embeddings = await self._embedding_service.generate_embeddings(texts_to_embed)

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                vector = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                # Store chunk_text_original as the "text" field for retrieval
                text_for_storage = chunk.get("chunk_text_original", "")
                meta = {
                    "source": filename,
                    "chunk_index": i,
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "page_number": chunk.get("page_number"),
                    "section_title": chunk.get("section_title"),
                    "language": chunk.get("language"),
                    "chunk_hash": chunk.get("chunk_hash"),
                }
                rows.append(
                    {
                        "id": f"{filename}#{i}",
                        "doc_id": filename,
                        "text": text_for_storage,
                        "vector": vector,
                        "metadata": json.dumps(meta),
                    }
                )

        if not rows:
            logger.warning("No chunks to add")
            return

        typed_batch = pa.Table.from_pylist(rows, schema=_LANCEDB_SCHEMA)
        if self._table is not None:
            self._table.add(typed_batch)
            logger.info("Added %d rows to LanceDB table '%s'", len(rows), _TABLE_NAME)
        else:
            self._table = self._db.create_table(
                _TABLE_NAME, data=typed_batch, schema=_LANCEDB_SCHEMA
            )
            logger.info("Created LanceDB table '%s' with %d rows", _TABLE_NAME, len(rows))
