"""
RagRetriever — vector search, graph-context search, and response generation.

Replaces the retrieval logic from the monolithic RagService.
Graph context is served by graph_engine.store.LanceDBGraphStore (local triples),
replacing the Microsoft GraphRAG CLI subprocess.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_OLLAMA_INDEX_URL = os.environ["OLLAMA_INDEX_URL"]
_OLLAMA_CHAT_URL = os.environ["OLLAMA_CHAT_URL"]
_CHAT_MODEL = os.environ["CHAT_MODEL"]
_EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
_TOP_K = int(os.environ.get("TOP_K", "5"))

_TABLE_NAME = "documents"


class RagRetriever:
    """
    Hybrid retrieval: vector similarity via LanceDB + graph context via graph_engine.

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
        else:
            self._ragdata_dir = self._root_dir / "ragdata"

        lance_path = (
            Path(lance_db_path) if lance_db_path else self._ragdata_dir / "lancedb"
        )
        self._lance_path = lance_path

        # --- Semantic Kernel setup ---
        from semantic_kernel import Kernel
        from semantic_kernel.connectors.ai.ollama import (
            OllamaChatCompletion,
            OllamaTextEmbedding,
        )

        self._kernel = Kernel()

        self._chat_service = OllamaChatCompletion(
            ai_model_id=_CHAT_MODEL,
            host=_OLLAMA_CHAT_URL,
            service_id="chat",
        )
        # Query-time embeddings come from the chat server (local, fast)
        self._query_embedding_service = OllamaTextEmbedding(
            ai_model_id=_EMBEDDING_MODEL,
            host=_OLLAMA_CHAT_URL,
            service_id="query_embedding",
        )

        self._kernel.add_service(self._chat_service)
        self._kernel.add_service(self._query_embedding_service)

        # --- LanceDB setup ---
        import lancedb  # noqa: PLC0415

        lance_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(lance_path))

        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)
            logger.info("Opened existing LanceDB table '%s'", _TABLE_NAME)
        else:
            self._table = None
            logger.info("LanceDB table '%s' not yet created", _TABLE_NAME)

    # ------------------------------------------------------------------
    # Lazy table accessor
    # ------------------------------------------------------------------

    def _ensure_table(self) -> bool:
        """Open the LanceDB table if it was created after this retriever was initialised.

        Returns ``True`` if the table is (now) available, ``False`` otherwise.
        """
        if self._table is not None:
            return True
        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)
            logger.info("Lazily opened LanceDB table '%s'", _TABLE_NAME)
            return True
        return False

    def reload_table(self) -> None:
        """Reopen the LanceDB table reference so newly indexed rows are visible.

        Call this after :meth:`RagIndexer.index_documents` completes to ensure
        subsequent searches see the latest data without a server restart.
        """
        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)
            logger.info("Reloaded LanceDB table '%s'", _TABLE_NAME)
        else:
            self._table = None
            logger.info("LanceDB table '%s' not yet created", _TABLE_NAME)

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    async def vector_search(self, query: str, top_k: int = _TOP_K) -> str:
        """Pure semantic similarity search — no graph context."""
        if not self._ensure_table():
            return "No documents indexed yet. Call index_documents() first."
        context = await self._raw_vector_context(query, top_k)
        return await self._generate_response(query, context)

    async def global_search(self, query: str) -> str:
        """Global knowledge-graph search — broad entity/relationship summaries."""
        graph_ctx = await asyncio.to_thread(self._graph_context, query)
        if not graph_ctx:
            return "No graph triples found for this query."
        return await self._generate_response(query, graph_ctx)

    async def hybrid_search(self, query: str, top_k: int = _TOP_K) -> str:
        """
        Parallel vector + graph search, merged context, SK-generated response.
        """
        vector_task = asyncio.create_task(self._raw_vector_context(query, top_k))
        graph_task = asyncio.create_task(asyncio.to_thread(self._graph_context, query))

        vector_ctx, graph_ctx = await asyncio.gather(vector_task, graph_task)

        merged = ""
        if vector_ctx:
            merged += f"Vector Search Results:\n{vector_ctx}"
        if graph_ctx:
            if merged:
                merged += "\n\n"
            merged += f"Graph Search Results:\n{graph_ctx}"

        if not merged:
            return "No relevant context found."

        return await self._generate_response(query, merged)

    def get_document_chunks(self, filename: str) -> list[dict]:
        """Return all chunks stored in LanceDB for *filename*, sorted by chunk_index."""
        if not self._ensure_table():
            return []
        try:
            df = self._table.to_pandas()
            rows = df[df["doc_id"] == filename]
            chunks = []
            for _, row in rows.iterrows():
                try:
                    meta = json.loads(row.get("metadata", "{}"))
                except (json.JSONDecodeError, TypeError):
                    meta = {}
                raw_text = row.get("text")
                chunks.append(
                    {
                        "chunk_index": meta.get("chunk_index", 0),
                        "text": raw_text if isinstance(raw_text, str) else "",
                        "chunk_type": meta.get("chunk_type"),
                        "page_number": meta.get("page_number"),
                        "section_title": meta.get("section_title"),
                        "language": meta.get("language"),
                    }
                )
            return sorted(chunks, key=lambda c: c["chunk_index"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to get chunks for '%s': %s", filename, exc)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _raw_vector_context(self, query: str, top_k: int = _TOP_K) -> str:
        """Return concatenated text chunks from LanceDB for *query*."""
        results = await self._raw_vector_results(query, top_k)
        return "\n\n".join(r["text"] for r in results)

    async def _raw_vector_results(self, query: str, top_k: int = _TOP_K) -> list[dict]:
        """Return raw LanceDB rows for *query*, merging vector and title-match results."""
        if not self._ensure_table():
            return []
        query_emb = await self._query_embedding_service.generate_embeddings([query])
        vector = (
            query_emb[0].tolist()
            if hasattr(query_emb[0], "tolist")
            else list(query_emb[0])
        )
        vector_results = self._table.search(vector).limit(top_k).to_list()
        title_results = self._title_search_results(query)
        seen_ids: set[str] = {r["id"] for r in vector_results}
        for row in title_results:
            if row.get("id") not in seen_ids:
                vector_results.append(row)
                seen_ids.add(row.get("id"))
        return vector_results

    def _title_search_results(self, query: str) -> list[dict]:
        """Return LanceDB rows whose ``section_title`` contains any keyword from *query*."""
        if not self._ensure_table():
            return []
        try:
            keywords = [w.lower() for w in query.split() if len(w) > 2]
            if not keywords:
                return []
            df = self._table.to_pandas()

            def _title_matches(metadata_str: str) -> bool:
                try:
                    meta = json.loads(metadata_str)
                    title = (meta.get("section_title") or "").lower()
                    return any(kw in title for kw in keywords)
                except (json.JSONDecodeError, TypeError):
                    return False

            mask = df["metadata"].apply(_title_matches)
            matched = df[mask]
            logger.debug("Title search found %d rows for query '%s'", len(matched), query)
            return matched.to_dict(orient="records")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Title search failed: %s", exc)
            return []

    def _graph_context(self, query: str) -> str:
        """
        Use spaCy NER to extract entities from *query*, look up their graph triples
        from LanceDBGraphStore, and format them for context injection.

        Format per triple:
            {subject} [{subject_type}] —{predicate}→ {object} [{object_type}] (weight={w:.2f})
        """
        try:
            from graph_engine.store import LanceDBGraphStore
        except ImportError:
            logger.debug("graph_engine not available, skipping graph context")
            return ""

        try:
            import spacy  # noqa: PLC0415
            _spacy_model = "en_core_web_sm"
            _local_model = Path(__file__).resolve().parent / "models" / _spacy_model
            nlp = spacy.load(_local_model if _local_model.exists() else _spacy_model)
        except Exception as exc:
            logger.debug("spaCy NER not available: %s", exc)
            # Fallback: use the raw query words as entity hints
            entities: list[str] = [w for w in query.split() if len(w) > 3]
        else:
            doc = nlp(query)
            entities = [ent.text for ent in doc.ents]
            if not entities:
                # Use noun chunks as fallback entity hints
                entities = [chunk.text for chunk in doc.noun_chunks if len(chunk.text) > 3]

        if not entities:
            return ""

        try:
            store = LanceDBGraphStore(self._lance_path)
        except Exception as exc:
            logger.debug("Failed to open graph store: %s", exc)
            return ""

        lines: list[str] = []
        seen: set[str] = set()

        for entity in entities:
            # Search by subject AND object to catch all relevant triples
            try:
                candidate_triples = (
                    store.get_triples(subject_name=entity)
                    + store.get_triples(object_name=entity)
                )
            except Exception as exc:
                logger.debug("Graph triple lookup failed for '%s': %s", entity, exc)
                continue
            for triple in candidate_triples:
                key = triple.get("triple_id", "")
                if key in seen:
                    continue
                seen.add(key)
                # get_triples returns nested dicts via _row_to_dict: triple["subject"]["name"]
                subj = triple.get("subject", {}).get("name", "")
                subj_t = triple.get("subject", {}).get("type", "")
                pred = triple.get("predicate", "")
                obj = triple.get("object", {}).get("name", "")
                obj_t = triple.get("object", {}).get("type", "")
                w = triple.get("weight", 1.0)
                lines.append(
                    f"{subj} [{subj_t}] —{pred}→ {obj} [{obj_t}] (weight={w:.2f})"
                )

        return "\n".join(lines)

    async def _generate_response(self, query: str, context: str) -> str:
        """Generate a final answer via the SK chat service given *context*."""
        from semantic_kernel.contents import ChatHistory  # noqa: PLC0415
        from semantic_kernel.connectors.ai.prompt_execution_settings import (
            PromptExecutionSettings,
        )

        history = ChatHistory()
        history.add_system_message(
            "You are a helpful assistant. Answer the user's question using only "
            "the provided context. If the context does not contain enough information, "
            "say so."
        )
        history.add_user_message(f"Context:\n{context}\n\nQuestion: {query}")
        settings = PromptExecutionSettings()
        responses = await self._chat_service.get_chat_message_contents(
            history, settings=settings
        )
        return str(responses[0]) if responses else ""
