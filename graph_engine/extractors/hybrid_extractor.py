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

    # Labels that produce noise on markdown/table content — skip these
    _SKIP_LABELS = frozenset({
        "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT",
    })

    # Max pairs to send to LLM — hard cap prevents runaway costs
    _MAX_TOTAL_PAIRS = 20

    @staticmethod
    def _is_garbage_entity(name: str) -> bool:
        """Return True if the entity text is obviously not a real entity."""
        stripped = name.strip()
        if len(stripped) < 3:
            return True
        # Markdown artifacts: headers, table separators, formatting
        if stripped.startswith("#") or stripped.startswith("|"):
            return True
        if "\n" in stripped:
            return True
        # Mostly punctuation / symbols
        alpha_chars = sum(1 for c in stripped if c.isalpha())
        if alpha_chars < 2:
            return True
        return False

    def extract(self, text: str, source_doc: str, on_batch: "Callable[[int, int], None] | None" = None) -> list[Triple]:
        if not text.strip():
            return []

        # Step 1 — NER: find entity candidates (filter before label mapping)
        nlp = _load_model(self._spacy._model_name)
        doc = nlp(text)

        # Pre-filter raw spaCy entities — skip noisy labels
        filtered_ents = [
            ent for ent in doc.ents
            if ent.label_ not in self._SKIP_LABELS
        ]

        # Build entity map only from filtered ents
        entity_map = self._spacy._build_entity_map(doc)
        entities = list({node.id: node for node in entity_map.values()}.values())

        # Remove garbage text (markdown, symbols, short strings)
        entities = [e for e in entities if not self._is_garbage_entity(e.name)]
        good_ids = {e.id for e in entities}
        entity_map = {k: v for k, v in entity_map.items() if v.id in good_ids}

        if len(entities) < self._fallback_threshold:
            logger.debug(
                "HybridExtractor: only %d entities after filtering, using LLM fallback", len(entities)
            )
            return self._llm_fallback.extract(text, source_doc)

        # Step 2 — build entity pairs from sentence co-occurrence only
        # Entities only pair if they appear in the same sentence — prevents
        # combinatorial explosion (N entities → N*(N-1)/2 pairs across chunk).
        pairs: list[tuple[Node, Node]] = []
        seen: set[tuple[str, str]] = set()
        for sent in doc.sents:
            sent_nodes: list[Node] = []
            for token in sent:
                node = entity_map.get(token.i)
                if node is not None and node.id not in {n.id for n in sent_nodes}:
                    sent_nodes.append(node)
            for i, a in enumerate(sent_nodes):
                for b in sent_nodes[i + 1 :]:
                    key = (min(a.id, b.id), max(a.id, b.id))
                    if key not in seen:
                        seen.add(key)
                        pairs.append((a, b))

        if not pairs:
            return []

        # Hard cap: take only the first N pairs to prevent runaway LLM calls
        if len(pairs) > self._MAX_TOTAL_PAIRS:
            logger.debug(
                "HybridExtractor: capping %d pairs to %d", len(pairs), self._MAX_TOTAL_PAIRS
            )
            pairs = pairs[: self._MAX_TOTAL_PAIRS]

        logger.debug(
            "HybridExtractor: %d entities → %d pairs", len(entities), len(pairs)
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

        return self._parse_classified(raw, source_doc)

    def _parse_classified(self, raw: str, source_doc: str) -> list[Triple]:
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

        triples: list[Triple] = []
        for item in items:
            try:
                subj = Node(
                    name=str(item["subject"]).strip(),
                    type=str(item.get("subject_type", "Other")),
                    source_doc=source_doc,
                )
                obj = Node(
                    name=str(item["object"]).strip(),
                    type=str(item.get("object_type", "Other")),
                    source_doc=source_doc,
                )
                relation = str(item.get("relation", "related_to")).strip()
                confidence = float(item.get("confidence", 1.0))

                triples.append(
                    Triple(
                        subject=subj,
                        predicate=relation,
                        object=obj,
                        weight=min(max(confidence, 0.0), 1.0),
                        source_doc=source_doc,
                    )
                )
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping malformed item: %s — %s", item, exc)

        return triples
