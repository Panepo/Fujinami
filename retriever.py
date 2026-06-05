"""
RagRetriever — vector search, graph-context search, and response generation.

Replaces the retrieval logic from the monolithic RagService.
Graph context is served by graph_engine.store.LanceDBGraphStore (local triples),
replacing the Microsoft GraphRAG CLI subprocess.

LLM stack: langchain-ollama (ChatOllama + OllamaEmbeddings).
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


def _format_triple(triple: dict) -> str:
    """Format a graph triple dict as a human-readable context line."""
    subj = triple.get("subject", {}).get("name", "")
    subj_t = triple.get("subject", {}).get("type", "")
    pred = triple.get("predicate", "")
    obj = triple.get("object", {}).get("name", "")
    obj_t = triple.get("object", {}).get("type", "")
    w = triple.get("weight", 1.0)
    return f"{subj} [{subj_t}] —{pred}→ {obj} [{obj_t}] (weight={w:.2f})"


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

        # --- langchain-ollama setup ---
        from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: PLC0415

        self._chat_service = ChatOllama(
            model=_CHAT_MODEL,
            base_url=_OLLAMA_CHAT_URL,
        )
        # Query-time embeddings come from the chat server (local, fast)
        self._query_embedding_service = OllamaEmbeddings(
            model=_EMBEDDING_MODEL,
            base_url=_OLLAMA_CHAT_URL,
        )

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
        """Open the LanceDB table if it was created after this retriever was initialised."""
        if self._table is not None:
            return True
        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)
            logger.info("Lazily opened LanceDB table '%s'", _TABLE_NAME)
            return True
        return False

    def reload_table(self) -> None:
        """Reopen the LanceDB table reference so newly indexed rows are visible."""
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
        """Parallel vector + graph search, merged context, LLM-generated response."""
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
        safe_id = filename.replace("'", "''")
        rows = (
            self._table
            .search(None)
            .where(f"doc_id = '{safe_id}'")
            .select(["text", "metadata"])
            .to_list()
        )
        chunks = []
        for row in rows:
            try:
                meta = json.loads(row.get("metadata") or "{}")
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
        # OllamaEmbeddings.embed_query() returns list[float] directly — no adapter needed
        vector = await asyncio.to_thread(
            self._query_embedding_service.embed_query, query
        )
        vector_results = self._table.search(vector).limit(top_k).to_list()
        return vector_results

    async def _raw_vector_results_from_embedding(
        self, embedding: list[float], top_k: int = _TOP_K
    ) -> list[dict]:
        """Return raw LanceDB rows using a pre-computed *embedding* vector (e.g. from HyDE)."""
        if not self._ensure_table():
            return []
        return self._table.search(embedding).limit(top_k).to_list()

    def _graph_context(self, query: str) -> str:
        """
        Build graph context for *query* using three cascading strategies:

        1. spaCy NER / noun-chunk extraction → LIKE-based triple lookup
           (normalized, case-insensitive, substring match).
        2. Raw query tokens → LIKE-based lookup (fallback when NER finds nothing).
        3. Embedding similarity → find stored entity names closest to the query
           and fetch their triples (fallback when strategies 1 & 2 yield nothing).

        Format per triple:
            {subject} [{subject_type}] —{predicate}→ {object} [{object_type}] (weight={w:.2f})
        """
        try:
            from graph_engine.store import LanceDBGraphStore, normalize_name  # noqa: PLC0415
        except ImportError:
            logger.debug("graph_engine not available, skipping graph context")
            return ""

        # ------------------------------------------------------------------
        # Entity extraction
        # ------------------------------------------------------------------
        try:
            import spacy  # noqa: PLC0415
            _spacy_model = "en_core_web_sm"
            _local_model = Path(__file__).resolve().parent / "models" / _spacy_model
            nlp = spacy.load(_local_model if _local_model.exists() else _spacy_model)
        except Exception as exc:
            logger.debug("spaCy NER not available: %s", exc)
            entities: list[str] = [
                normalize_name(w) for w in query.split() if len(w) > 3
            ]
        else:
            doc = nlp(query)
            entities = [normalize_name(ent.text) for ent in doc.ents]
            if not entities:
                entities = [
                    normalize_name(chunk.text)
                    for chunk in doc.noun_chunks
                    if len(chunk.text) > 3
                ]
            # Strategy 2 fallback: also include raw tokens so short queries still hit
            token_entities = [normalize_name(w) for w in query.split() if len(w) > 3]
            # Merge, preserving order, no duplicates
            seen_ents: set[str] = set(entities)
            for tok in token_entities:
                if tok not in seen_ents:
                    entities.append(tok)
                    seen_ents.add(tok)

        try:
            store = LanceDBGraphStore(self._lance_path)
        except Exception as exc:
            logger.debug("Failed to open graph store: %s", exc)
            return ""

        lines: list[str] = []
        seen: set[str] = set()

        # ------------------------------------------------------------------
        # Strategy 1 & 2: LIKE-based lookup for every extracted entity
        # ------------------------------------------------------------------
        for entity in entities:
            if not entity:
                continue
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
                lines.append(_format_triple(triple))

        # ------------------------------------------------------------------
        # Strategy 3: embedding-based entity lookup (only when LIKE found nothing)
        # ------------------------------------------------------------------
        if not lines:
            try:
                all_names = store.get_all_entity_names()
                if all_names:
                    import numpy as np  # noqa: PLC0415

                    q_vec = self._query_embedding_service.embed_query(query)
                    e_vecs = self._query_embedding_service.embed_documents(all_names)
                    q_arr = np.array(q_vec, dtype=float)
                    e_arr = np.array(e_vecs, dtype=float)
                    norms = np.linalg.norm(e_arr, axis=1) * np.linalg.norm(q_arr) + 1e-9
                    sims = (e_arr @ q_arr) / norms
                    top_idxs = np.argsort(sims)[::-1][:5]
                    for idx in top_idxs:
                        if float(sims[idx]) < 0.5:
                            break
                        matched = all_names[int(idx)]
                        logger.debug(
                            "Embedding entity match: '%s' (sim=%.3f)", matched, sims[idx]
                        )
                        for triple in (
                            store.get_triples(subject_name=matched)
                            + store.get_triples(object_name=matched)
                        ):
                            key = triple.get("triple_id", "")
                            if key in seen:
                                continue
                            seen.add(key)
                            lines.append(_format_triple(triple))
            except Exception as exc:
                logger.debug("Embedding-based entity lookup failed: %s", exc)

        return "\n".join(lines)

    async def _generate_response(self, query: str, context: str) -> str:
        """Generate a final answer via ChatOllama given *context*."""
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

        messages = [
            SystemMessage(content=(
                "You are a helpful assistant. Answer the user's question using only "
                "the provided context. If the context does not contain enough information, "
                "say so."
            )),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        try:
            response = await self._chat_service.ainvoke(messages)
            return response.content or ""
        except Exception as exc:
            logger.warning("_generate_response error: %s", exc)
            return ""


