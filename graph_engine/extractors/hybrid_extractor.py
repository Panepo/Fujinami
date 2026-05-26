"""
HybridExtractor — Option B: spaCy finds entities, LLM classifies relations.

How it works
------------
1. SpacyExtractor runs NER to find entity candidates (fast, ~50ms)
2. Entity pairs from spaCy are passed to the LLM in a *shorter* prompt
   that only asks "what relation exists between these two entities?"
3. This is ~3× cheaper per chunk than asking the LLM to extract
   everything from scratch.

Fallback behavior
-----------------
- If spaCy finds fewer than 2 entities in a chunk, the full LLMExtractor
  is used on that chunk (handles implicit entities the NER missed).
- The ``fallback_threshold`` parameter controls this.

Speed: ~100–500ms per chunk (spaCy entity step) + short LLM call per pair.
Quality: close to full LLM, but cheaper.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from graph_engine.base import BaseExtractor
from graph_engine.extractors.llm_extractor import LLMExtractor
from graph_engine.extractors.spacy_extractor import SpacyExtractor, _load_model
from graph_engine.models import Node, Triple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Relation-only prompt
# ---------------------------------------------------------------------------

_RELATION_SYSTEM = """\
You are a knowledge-graph relation classifier.
Given a text passage and a list of entity pairs, output the relation
between each pair.

Relation types MUST be one of:
  is_a, part_of, causes, governs, opposes, related_to,
  supports_protocol, certified_by

If no meaningful relation exists between a pair, use "related_to".
Output ONLY valid JSON — an array. No explanation, no markdown fences.

Output format:
[
  {
    "subject": "entity name",
    "subject_type": "EntityType",
    "relation": "relation_type",
    "object": "entity name",
    "object_type": "EntityType",
    "confidence": 0.0-1.0
  }
]
"""

_RELATION_USER = """\
Text:
{text}

Entity pairs to classify:
{pairs}
"""


class HybridExtractor(BaseExtractor):
    """
    spaCy NER → LLM relation classification.

    Parameters
    ----------
    spacy_model:
        spaCy model name for NER (``en_core_web_sm`` or ``en_core_web_trf``).
    ollama_url:
        Ollama server URL. Defaults to ``OLLAMA_INDEX_URL`` env var.
    model:
        LLM model name. Defaults to ``EXTRACT_MODEL`` env var.
    fallback_threshold:
        Minimum number of entities spaCy must find before skipping
        the full LLM fallback. If spaCy finds fewer entities, the full
        LLMExtractor is used on that chunk.
    max_pairs_per_call:
        Batch entity pairs into groups of this size to keep prompts short.
    timeout:
        HTTP timeout for Ollama calls.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        ollama_url: str | None = None,
        model: str | None = None,
        fallback_threshold: int = 2,
        max_pairs_per_call: int = 10,
        timeout: float = 120.0,
    ) -> None:
        self._spacy = SpacyExtractor(model_name=spacy_model)
        self._llm_fallback = LLMExtractor(
            ollama_url=ollama_url, model=model, timeout=timeout
        )
        self._fallback_threshold = fallback_threshold
        self._max_pairs = max_pairs_per_call
        self._url = (ollama_url or os.environ.get("OLLAMA_INDEX_URL", "")).rstrip("/")
        self._model = model or os.environ.get("EXTRACT_MODEL", "granite4.1:8b")
        self._timeout = timeout

    # Max pairs to send to LLM — hard cap prevents runaway costs
    _MAX_TOTAL_PAIRS = 20

    def extract(self, text: str, source_doc: str, on_batch: "Callable[[int, int], None] | None" = None) -> list[Triple]:
        if not text.strip():
            return []

        # Step 1 — Run spaCy's full dependency-based extraction.
        # This reuses SpacyExtractor's NER + dep-parse pipeline which only
        # produces nodes that are syntactic subjects/objects of verbs —
        # naturally filtering out markdown artifacts and noun-chunk noise.
        nlp = _load_model(self._spacy._model_name)
        doc = nlp(text)
        entity_map = self._spacy._build_entity_map(doc)
        spacy_triples = self._spacy._extract_triples(doc, entity_map, source_doc)

        if len(spacy_triples) < self._fallback_threshold:
            logger.debug(
                "HybridExtractor: only %d spaCy triples, using LLM fallback", len(spacy_triples)
            )
            return self._llm_fallback.extract(text, source_doc)

        # Step 2 — Derive pairs from spaCy's clean dependency triples.
        # Using dep-parsed pairs instead of co-occurrence prevents garbage
        # noun chunks from being paired and sent to the LLM.
        seen: set[tuple[str, str]] = set()
        pairs: list[tuple[Node, Node]] = []
        for t in spacy_triples:
            key = (min(t.subject.id, t.object.id), max(t.subject.id, t.object.id))
            if key not in seen:
                seen.add(key)
                pairs.append((t.subject, t.object))

        # Hard cap: take only the first N pairs to prevent runaway LLM calls
        if len(pairs) > self._MAX_TOTAL_PAIRS:
            logger.debug(
                "HybridExtractor: capping %d pairs to %d", len(pairs), self._MAX_TOTAL_PAIRS
            )
            pairs = pairs[: self._MAX_TOTAL_PAIRS]

        logger.debug(
            "HybridExtractor: %d spaCy triples → %d unique pairs", len(spacy_triples), len(pairs)
        )

        # Step 3 — classify relations in batches
        triples: list[Triple] = []
        total_batches = (len(pairs) + self._max_pairs - 1) // self._max_pairs
        for batch_idx, batch_start in enumerate(range(0, len(pairs), self._max_pairs)):
            if on_batch:
                on_batch(batch_idx + 1, total_batches)
            batch = pairs[batch_start : batch_start + self._max_pairs]
            triples.extend(self._classify_relations(text, batch, source_doc))

        return triples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_relations(
        self, text: str, pairs: list[tuple[Node, Node]], source_doc: str
    ) -> list[Triple]:
        # Build a case-insensitive lookup so _parse_classified can reuse spaCy nodes
        node_lookup: dict[str, Node] = {}
        for a, b in pairs:
            node_lookup[a.name.lower()] = a
            node_lookup[b.name.lower()] = b

        pairs_json = json.dumps(
            [
                {"subject": a.name, "subject_type": a.type, "object": b.name, "object_type": b.type}
                for a, b in pairs
            ],
            indent=2,
        )

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _RELATION_SYSTEM},
                {
                    "role": "user",
                    "content": _RELATION_USER.format(
                        text=text[:3000], pairs=pairs_json
                    ),
                },
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self._url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode())
                raw = body.get("message", {}).get("content", "")
        except Exception as exc:
            logger.warning("HybridExtractor Ollama call failed: %s", exc)
            return []

        return self._parse_classified(raw, source_doc, node_lookup)

    def _parse_classified(self, raw: str, source_doc: str, node_lookup: dict[str, "Node"] | None = None) -> list[Triple]:
        if not raw:
            return []

        clean = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()

        try:
            items: list[dict[str, Any]] = json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", clean, re.DOTALL)
            if not match:
                return []
            try:
                items = json.loads(match.group())
            except json.JSONDecodeError:
                return []

        node_lookup = node_lookup or {}
        triples: list[Triple] = []
        for item in items:
            try:
                subj_name = str(item["subject"]).strip()
                obj_name = str(item["object"]).strip()

                # Reuse spaCy node if name matches (case-insensitive); fall back to LLM's data
                spacy_subj = node_lookup.get(subj_name.lower())
                spacy_obj = node_lookup.get(obj_name.lower())

                subj = (
                    spacy_subj.model_copy(update={"source_doc": source_doc})
                    if spacy_subj is not None
                    else Node(name=subj_name, type=str(item.get("subject_type", "Other")), source_doc=source_doc)
                )
                obj = (
                    spacy_obj.model_copy(update={"source_doc": source_doc})
                    if spacy_obj is not None
                    else Node(name=obj_name, type=str(item.get("object_type", "Other")), source_doc=source_doc)
                )
                relation = str(item.get("relation", "related_to")).strip()
                confidence = float(item.get("confidence", 1.0))

                triples.append(
                    Triple(
                        subject=subj,
                        object=obj,
                        predicate=relation,
                        weight=min(max(confidence, 0.0), 1.0),
                        source_doc=source_doc,
                    )
                )
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping malformed item: %s — %s", item, exc)

        return triples
