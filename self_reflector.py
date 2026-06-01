"""
Self-RAG reflection loop for Fujinami.

Wraps the existing retrieval pipeline with four LLM-gated decisions:
1. Is retrieval needed?
2. Which retrieved chunks are actually relevant?
3. Is the generated answer grounded in the context?
4. If not grounded, refine the query and retry (up to max_iterations).
"""
from __future__ import annotations

import json
import logging

from models import SelfRagMeta, SourceChunk

_logger = logging.getLogger(__name__)

_RETRIEVAL_NEEDED_PROMPT = (
    'Does the following question require looking up specific documents or data to answer '
    'accurately? Answer with a single JSON object: {{"needed": true}} or {{"needed": false}}.\n\n'
    "Question: {query}"
)

_RELEVANCE_PROMPT = (
    "Given the question below, decide whether the following text excerpt is relevant "
    "to answering it. Reply ONLY with a JSON object: "
    '{{"relevant": true}} or {{"relevant": false}}.\n\n'
    "Question: {query}\n\nExcerpt:\n{excerpt}"
)

_GROUNDING_PROMPT = (
    "Given the context and the answer below, decide whether the answer is fully supported "
    "by the context without adding unsupported information. "
    'Reply ONLY with a JSON object: {{"grounded": true}} or {{"grounded": false}}.\n\n'
    "Context:\n{context}\n\nAnswer:\n{answer}"
)

_REFINE_QUERY_PROMPT = (
    "The following question was not answered satisfactorily from the retrieved documents. "
    "Rewrite it to be more specific so that a document search would return better results. "
    "Reply with only the rewritten query as plain text.\n\n"
    "Original question: {query}"
)


class SelfReflector:
    """Applies a self-RAG loop around the RAG service retrieval pipeline."""

    def __init__(self, rag_service, max_iterations: int = 2) -> None:
        self._rag = rag_service
        self._max_iterations = max_iterations

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def query(
        self,
        query: str,
        method: str,
        top_k: int,
    ) -> tuple[str, list[SourceChunk] | None, str | None, SelfRagMeta]:
        """Run the self-RAG loop and return ``(answer, sources, graphrag_context, meta)``."""

        # Step 1 — decide if retrieval is needed
        needed = await self._should_retrieve(query)

        if not needed:
            answer = await self._rag._generate_response(query, "")
            meta = SelfRagMeta(needed=False, relevant_chunks=0, grounded=True, iterations=0)
            return answer, None, None, meta

        # Retrieval + reflection loop
        current_query = query
        iterations = 0
        sources: list[SourceChunk] = []
        graphrag_context: str | None = None
        answer = ""
        relevant_chunks = 0

        while iterations < self._max_iterations:
            iterations += 1

            # Fetch raw results
            raw_rows: list[dict] = []
            vector_context: str | None = None
            if method != "graph":
                raw_rows = await self._rag._raw_vector_results(current_query, top_k)
                vector_context = "\n\n".join(r.get("text", "") for r in raw_rows)

            if method in ("graph", "hybrid"):
                graphrag_context = await self._rag._graphrag_search(current_query, method="local")

            # Build SourceChunk list from raw rows
            all_sources: list[SourceChunk] = []
            for row in raw_rows:
                try:
                    meta_json = json.loads(row.get("metadata", "{}"))
                except (json.JSONDecodeError, TypeError):
                    meta_json = {}
                all_sources.append(
                    SourceChunk(
                        doc_id=row.get("doc_id", ""),
                        chunk_index=meta_json.get("chunk_index", 0),
                        excerpt=row.get("text", "")[:200],
                        full_text=row.get("text", ""),
                    )
                )

            # Step 2 — filter to relevant chunks (batched single call)
            if all_sources:
                relevant_sources = await self._filter_relevant(current_query, all_sources)
            else:
                relevant_sources = []

            relevant_chunks = len(relevant_sources)
            sources = relevant_sources if relevant_sources else all_sources

            # Build merged context from relevant chunks + graph
            parts: list[str] = []
            if relevant_sources:
                parts.append("\n\n".join(s.full_text for s in relevant_sources))
            elif vector_context:
                parts.append(vector_context)
            if graphrag_context and method != "vector":
                parts.append(f"Graph context:\n{graphrag_context}")

            merged_context = "\n\n".join(parts)

            # Generate answer
            answer = await self._rag._generate_response(current_query, merged_context)

            # Step 3 — check grounding
            grounded = await self._check_grounding(current_query, answer, merged_context)

            if grounded:
                meta = SelfRagMeta(
                    needed=True,
                    relevant_chunks=relevant_chunks,
                    grounded=True,
                    iterations=iterations,
                )
                return answer, sources, graphrag_context, meta

            # Grounding failed — refine query for next iteration
            if iterations < self._max_iterations:
                current_query = await self._refine_query(query)
                _logger.debug("Self-RAG: grounding failed, refined query: %s", current_query)

        # Exhausted iterations
        meta = SelfRagMeta(
            needed=True,
            relevant_chunks=relevant_chunks,
            grounded=False,
            iterations=iterations,
        )
        return answer, sources, graphrag_context, meta

    # ------------------------------------------------------------------
    # Internal LLM helpers
    # ------------------------------------------------------------------

    async def _llm_call(self, prompt: str) -> str:
        """Send *prompt* as a single user message and return the text reply."""
        from semantic_kernel.contents import ChatHistory  # noqa: PLC0415
        from semantic_kernel.connectors.ai.prompt_execution_settings import (  # noqa: PLC0415
            PromptExecutionSettings,
        )

        history = ChatHistory()
        history.add_user_message(prompt)
        responses = await self._rag._chat_service.get_chat_message_contents(
            history, settings=PromptExecutionSettings()
        )
        return str(responses[0]).strip() if responses else ""

    async def _should_retrieve(self, query: str) -> bool:
        prompt = _RETRIEVAL_NEEDED_PROMPT.format(query=query)
        try:
            raw = await self._llm_call(prompt)
            # Extract JSON from response (model may add surrounding text)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            data = json.loads(raw[start:end])
            return bool(data.get("needed", True))
        except Exception:
            _logger.debug("Self-RAG: _should_retrieve parse failed, defaulting to True")
            return True

    async def _filter_relevant(
        self, query: str, chunks: list[SourceChunk]
    ) -> list[SourceChunk]:
        """Batch all chunks into a single prompt and return only relevant ones."""
        items = "\n".join(
            f'[{i}] {c.excerpt}' for i, c in enumerate(chunks)
        )
        prompt = (
            f"Given the question below, identify which of the following numbered excerpts "
            f"are relevant to answering it. Reply ONLY with a JSON array of the relevant "
            f"indices, e.g. [0, 2]. Return [] if none are relevant.\n\n"
            f"Question: {query}\n\nExcerpts:\n{items}"
        )
        try:
            raw = await self._llm_call(prompt)
            start = raw.find("[")
            end = raw.rfind("]") + 1
            indices: list[int] = json.loads(raw[start:end])
            return [chunks[i] for i in indices if 0 <= i < len(chunks)]
        except Exception:
            _logger.debug("Self-RAG: _filter_relevant parse failed, returning all chunks")
            return chunks

    async def _check_grounding(self, query: str, answer: str, context: str) -> bool:
        prompt = _GROUNDING_PROMPT.format(context=context[:3000], answer=answer[:1500])
        try:
            raw = await self._llm_call(prompt)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            data = json.loads(raw[start:end])
            return bool(data.get("grounded", True))
        except Exception:
            _logger.debug("Self-RAG: _check_grounding parse failed, assuming grounded")
            return True

    async def _refine_query(self, query: str) -> str:
        prompt = _REFINE_QUERY_PROMPT.format(query=query)
        try:
            refined = await self._llm_call(prompt)
            return refined.strip() or query
        except Exception:
            return query
