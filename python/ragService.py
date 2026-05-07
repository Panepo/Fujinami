"""
RagService — hybrid RAG using Semantic Kernel + LanceDB + Microsoft GraphRAG.

Stack
-----
- LLM / Embeddings : Ollama (qwen3.6:9b / locusai/all-minilm-l6-v2:latest)
- Vector store      : LanceDB (persistent, embedded, file-based)
- Graph engine      : Microsoft GraphRAG (CLI subprocess)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Optional

import pyarrow as pa
from dotenv import load_dotenv

from document_loader import DocumentLoader, SUPPORTED_EXTENSIONS

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LanceDB schema
# ---------------------------------------------------------------------------

_LANCEDB_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 384)),
        pa.field("metadata", pa.string()),
    ]
)

_TABLE_NAME = "documents"
_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200
_TOP_K = 5

# ---------------------------------------------------------------------------
# Ollama service configuration
# ---------------------------------------------------------------------------

_OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
_CHAT_MODEL = os.environ["CHAT_MODEL"]
_EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
_VLM_MODEL = os.environ["VLM_MODEL"]


class RagService:
    """
    Hybrid retrieval-augmented generation service.

    Parameters
    ----------
    collection_name:
        Name of the document collection (e.g. ``"A"``, ``"B"``).
        When set, documents live in ``{root_dir}/data/{collection_name}/``
        and each collection gets its own isolated ragdata and LanceDB store
        under ``{root_dir}/ragdata/{collection_name}/``.
        When ``None`` (default), the legacy single-collection layout is used.
    root_dir:
        Root directory for data, ragdata, and LanceDB storage.
        Defaults to the directory containing this file (``python/``).
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

        # --- Semantic Kernel setup ---
        from semantic_kernel import Kernel
        from semantic_kernel.connectors.ai.ollama import (
            OllamaChatCompletion,
            OllamaTextEmbedding,
        )

        self._kernel = Kernel()

        self._chat_service = OllamaChatCompletion(
            ai_model_id=_CHAT_MODEL,
            host=_OLLAMA_BASE_URL,
            service_id="chat",
        )
        self._embedding_service = OllamaTextEmbedding(
            ai_model_id=_EMBEDDING_MODEL,
            host=_OLLAMA_BASE_URL,
            service_id="embedding",
        )

        self._kernel.add_service(self._chat_service)
        self._kernel.add_service(self._embedding_service)

        # --- LanceDB setup ---
        import lancedb  # noqa: PLC0415

        lance_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(lance_path))

        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)
            logger.info("Opened existing LanceDB table '%s'", _TABLE_NAME)
        else:
            self._table = None
            logger.info("LanceDB table '%s' will be created on first index", _TABLE_NAME)

        self._manifest_path = lance_path / "file_manifest.json"

        if collection_name is not None:
            self._ensure_settings_yaml()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_documents(self, documents_dir: str | Path | None = None) -> None:
        """
        Incremental indexing pipeline:

        1. Compute per-file delta against ``file_manifest.json``.
        2. If nothing changed, return immediately.
        3. Delete ``.txt`` files and LanceDB rows for removed/modified sources.
        4. Load and convert only new/modified files via :class:`DocumentLoader`.
        5. Write plain-text ``.txt`` files to ``{root_dir}/data/``.
        6. Run GraphRAG CLI indexer (subprocess).
        7. Chunk, embed, and add vectors into LanceDB.
        8. Save updated manifest.
        """
        if documents_dir is None:
            documents_dir = self._data_dir
        documents_dir = Path(documents_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Step 1 — delta detection
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

        # Step 3 — remove stale .txt files and LanceDB rows
        for doc_id in removed_sources:
            txt_path = self._data_dir / (Path(doc_id).stem + ".txt")
            if txt_path.exists():
                txt_path.unlink()
                logger.info("Removed stale txt: %s", txt_path.name)
        if removed_sources:
            self._remove_from_lancedb(list(removed_sources))

        if not changed_sources:
            # Only deletions — still need to re-index GraphRAG
            await self._run_graphrag_index()
            self._save_manifest(documents_dir)
            return

        # Step 4 — load only changed files
        loader = DocumentLoader(
            ollama_base_url=_OLLAMA_BASE_URL,
            vlm_model=_VLM_MODEL,
        )
        doc_texts = loader.load_directory(documents_dir, files_filter=changed_sources)

        if not doc_texts:
            logger.warning("No content loaded for changed sources")
            self._save_manifest(documents_dir)
            return

        # Step 5 — write .txt for changed files only
        for filename, text in doc_texts.items():
            txt_path = self._data_dir / (Path(filename).stem + ".txt")
            txt_path.write_text(text, encoding="utf-8")
            logger.info("Wrote %s", txt_path.name)

        # Step 6 — GraphRAG indexing
        await self._run_graphrag_index()

        # Step 7 — add new chunks to LanceDB
        await self._upsert_to_lancedb(doc_texts)

        # Step 8 — persist manifest
        self._save_manifest(documents_dir)

    def _ensure_settings_yaml(self) -> None:
        """Generate a per-collection ``settings.yaml`` from the root template.

        Reads ``{root_dir}/ragdata/settings.yaml`` as a template, sets
        ``input.base_dir`` to the absolute path of ``self._data_dir``, and
        writes the result to ``{ragdata_dir}/settings.yaml``.  A new file is
        written whenever the stored ``base_dir`` value differs from the
        current one.
        """
        import yaml  # noqa: PLC0415

        template_path = self._root_dir / "ragdata" / "settings.yaml"
        if not template_path.exists():
            logger.warning("Settings template not found at %s", template_path)
            return

        config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        target_base_dir = str(self._data_dir.resolve())

        dest_path = self._ragdata_dir / "settings.yaml"
        if dest_path.exists():
            existing = yaml.safe_load(dest_path.read_text(encoding="utf-8"))
            if existing.get("input", {}).get("base_dir") == target_base_dir:
                logger.info(
                    "settings.yaml for collection '%s' already up to date",
                    self._collection_name,
                )
                return

        config.setdefault("input", {})["base_dir"] = target_base_dir
        self._ragdata_dir.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")
        logger.info(
            "Wrote settings.yaml for collection '%s' → %s",
            self._collection_name,
            dest_path,
        )

    async def _run_graphrag_index(self) -> None:
        """Run ``graphrag index`` as a subprocess."""
        cmd = [sys.executable, "-m", "graphrag", "index", "--root", str(self._ragdata_dir)]
        logger.info("Running GraphRAG indexer: %s", " ".join(cmd))
        result = await asyncio.to_thread(
            subprocess.run,  # nosec B603
            cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning("GraphRAG indexing stderr:\n%s", result.stderr)
        else:
            logger.info("GraphRAG indexing completed successfully")

    def _load_manifest(self) -> dict[str, dict]:
        """Load ``file_manifest.json`` → ``{filename: {mtime, size}}``.

        Returns an empty dict if the file doesn't exist or is malformed.
        """
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
        stored_manifest: dict[str, dict],
    ) -> tuple[set[str], set[str], set[str], set[str]]:
        """Compare on-disk files against *stored_manifest*.

        Returns
        -------
        tuple of (new_files, modified_files, deleted_files, unchanged_files)
            Each element is a ``set[str]`` of filenames (basename only).
        """
        on_disk: dict[str, dict] = {}
        for file_path in documents_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                stat = file_path.stat()
                on_disk[file_path.name] = {"mtime": stat.st_mtime, "size": stat.st_size}

        on_disk_names = set(on_disk)
        stored_names = set(stored_manifest)

        new_files = on_disk_names - stored_names
        deleted_files = stored_names - on_disk_names
        modified_files: set[str] = set()
        unchanged_files: set[str] = set()

        for name in on_disk_names & stored_names:
            cur = on_disk[name]
            prev = stored_manifest[name]
            if cur["mtime"] != prev["mtime"] or cur["size"] != prev["size"]:
                modified_files.add(name)
            else:
                unchanged_files.add(name)

        return new_files, modified_files, deleted_files, unchanged_files

    def _save_manifest(self, documents_dir: Path) -> None:
        """Write fresh ``file_manifest.json`` reflecting all on-disk files."""
        manifest: dict[str, dict] = {}
        for file_path in documents_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                stat = file_path.stat()
                manifest[file_path.name] = {"mtime": stat.st_mtime, "size": stat.st_size}
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        logger.info("Manifest saved with %d entries", len(manifest))

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

    async def _upsert_to_lancedb(self, doc_texts: dict[str, str]) -> None:
        """Chunk texts, generate embeddings, and add rows into LanceDB.

        This method is pure-add; callers must remove stale rows beforehand
        via :meth:`_remove_from_lancedb`.
        """
        rows: list[dict] = []

        for filename, text in doc_texts.items():
            chunks = self._chunk_text(text, _CHUNK_SIZE, _CHUNK_OVERLAP)
            if not chunks:
                continue

            embeddings = await self._embedding_service.generate_embeddings(chunks)

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                vector = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                rows.append(
                    {
                        "id": f"{filename}#{i}",
                        "doc_id": filename,
                        "text": chunk,
                        "vector": vector,
                        "metadata": json.dumps(
                            {"source": filename, "chunk_index": i}
                        ),
                    }
                )

        if not rows:
            logger.warning("No chunks to add")
            return

        if self._table is not None:
            self._table.add(rows)
            logger.info("Added %d rows to LanceDB table '%s'", len(rows), _TABLE_NAME)
        else:
            self._table = self._db.create_table(
                _TABLE_NAME, data=rows, schema=_LANCEDB_SCHEMA
            )
            logger.info("Created LanceDB table '%s' with %d rows", _TABLE_NAME, len(rows))

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    async def vector_search(self, query: str, top_k: int = _TOP_K) -> str:
        """Pure semantic similarity search — no graph."""
        if self._table is None:
            return "No documents indexed yet. Call index_documents() first."

        context = await self._raw_vector_context(query, top_k)
        return await self._generate_response(query, context)

    async def global_search(self, query: str) -> str:
        """GraphRAG global search — broad community-level summaries."""
        return await self._graphrag_search(query, method="global")

    async def hybrid_search(self, query: str, top_k: int = _TOP_K) -> str:
        """
        Run SK vector search and GraphRAG local search in parallel,
        merge context, and generate a response via SK.
        """
        vector_task = asyncio.create_task(self._raw_vector_context(query, top_k))
        graphrag_task = asyncio.create_task(self._graphrag_search(query, method="local"))

        vector_ctx, graphrag_ctx = await asyncio.gather(vector_task, graphrag_task)

        merged = (
            f"Vector Search Results:\n{vector_ctx}\n\n"
            f"Graph Search Results:\n{graphrag_ctx}"
        )
        return await self._generate_response(query, merged)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _raw_vector_context(self, query: str, top_k: int = _TOP_K) -> str:
        """Return raw text chunks from LanceDB for *query*."""
        results = await self._raw_vector_results(query, top_k)
        return "\n\n".join(r["text"] for r in results)

    async def _raw_vector_results(self, query: str, top_k: int = _TOP_K) -> list[dict]:
        """Return raw LanceDB rows for *query* (keys: doc_id, text, metadata)."""
        if self._table is None:
            return []
        query_emb = await self._embedding_service.generate_embeddings([query])
        vector = (
            query_emb[0].tolist()
            if hasattr(query_emb[0], "tolist")
            else list(query_emb[0])
        )
        return self._table.search(vector).limit(top_k).to_list()

    async def _graphrag_search(self, query: str, method: str = "local") -> str:
        """Run a GraphRAG query subprocess and return its stdout."""
        cmd = [
            sys.executable, "-m", "graphrag", "query",
            "--root", str(self._ragdata_dir),
            "--method", method,
            "--query", query,
        ]
        result = await asyncio.to_thread(
            subprocess.run,  # nosec B603
            cmd,
            capture_output=True,
            text=True,
        )
        output = result.stdout.strip()
        if not output:
            output = result.stderr.strip()
        return output or f"(GraphRAG {method} search returned no output)"

    async def _generate_response(self, query: str, context: str) -> str:
        """Generate a final answer using the SK chat service given *context*."""
        from semantic_kernel.contents import ChatHistory  # noqa: PLC0415

        history = ChatHistory()
        history.add_system_message(
            "You are a helpful assistant. Answer the user's question using only "
            "the provided context. If the context does not contain enough information, "
            "say so."
        )
        history.add_user_message(
            f"Context:\n{context}\n\nQuestion: {query}"
        )
        responses = await self._chat_service.get_chat_message_contents(history)
        return str(responses[0]) if responses else ""

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
        """Split *text* into overlapping chunks of at most *size* characters."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += size - overlap
        return chunks


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_rag_service(
    collection_name: str,
    root_dir: str | Path | None = None,
) -> RagService:
    """Return a :class:`RagService` scoped to *collection_name*.

    Equivalent to ``RagService(collection_name=collection_name, root_dir=root_dir)``.

    Parameters
    ----------
    collection_name:
        Name of the document collection (e.g. ``"A"``, ``"B"``).
    root_dir:
        Root directory for data and ragdata storage.
        Defaults to the directory containing this file (``python/``).
    """
    return RagService(collection_name=collection_name, root_dir=root_dir)
