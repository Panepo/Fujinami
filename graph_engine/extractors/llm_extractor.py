"""
LLMExtractor — graph extraction via local Ollama (granite / llama / etc.).

Sends each text chunk to the Ollama chat API with a structured prompt
that enforces:
  - Closed entity type list (from graph-design-spec §3)
  - Closed relation set (from graph-design-spec §4)
  - JSON output format

Uses the same OLLAMA_INDEX_URL and EXTRACT_MODEL env vars already
configured in ragService.py — no new credentials needed.

Speed: ~0.5–2s per chunk on DGX GPU.
Quality: high — handles implicit relations, abbreviations, messy text.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Any

from graph_engine.base import BaseExtractor
from graph_engine.models import Node, Triple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a knowledge-graph extraction engine.
Your task: extract entity-relation-entity triples from the given text.

Rules:
1. Entity types MUST be one of:
   Person, Organization, Location, Date, Event, Concept,
   Device, Component, Specification, Protocol, Interface,
   Software, Standard, Error, Configuration, Product, Version, Vendor, Other

2. Relation types MUST be one of:
   is_a, part_of, causes, governs, opposes, related_to,
   supports_protocol, certified_by

3. Output ONLY valid JSON — an array of triple objects.
   Do NOT include explanation or markdown fences.

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

If no triples can be extracted, return an empty array: []
"""

_USER_TEMPLATE = "Extract triples from this text:\n\n{text}"


class LLMExtractor(BaseExtractor):
    """
    Extract triples by calling an Ollama-compatible LLM API.

    Parameters
    ----------
    ollama_url:
        Base URL of the Ollama server (e.g. ``http://172.16.7.52:11434``).
        Defaults to the ``OLLAMA_INDEX_URL`` env var.
    model:
        Model name to use (e.g. ``granite4.1:8b``).
        Defaults to the ``EXTRACT_MODEL`` env var.
    timeout:
        HTTP timeout in seconds for each LLM call.
    """

    def __init__(
        self,
        ollama_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._url = (ollama_url or os.environ.get("OLLAMA_INDEX_URL", "")).rstrip("/")
        self._model = model or os.environ.get("EXTRACT_MODEL", "granite4.1:8b")
        self._timeout = timeout

        if not self._url:
            raise ValueError("OLLAMA_INDEX_URL env var or ollama_url parameter is required")

    def extract(self, text: str, source_doc: str) -> list[Triple]:
        if not text.strip():
            return []

        raw = self._call_ollama(text)
        return self._parse_response(raw, source_doc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_ollama(self, text: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(text=text[:4000])},
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
                return body.get("message", {}).get("content", "")
        except Exception as exc:
            logger.warning("Ollama call failed: %s", exc)
            return ""

    def _parse_response(self, raw: str, source_doc: str) -> list[Triple]:
        if not raw:
            return []

        # Strip any accidental markdown fences
        clean = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()

        try:
            items: list[dict[str, Any]] = json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract JSON array from within the response
            match = re.search(r"\[.*\]", clean, re.DOTALL)
            if not match:
                logger.warning("LLMExtractor: could not parse JSON from response")
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
                    type=str(item.get("subject_type", "Other")).strip(),
                    source_doc=source_doc,
                )
                obj = Node(
                    name=str(item["object"]).strip(),
                    type=str(item.get("object_type", "Other")).strip(),
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
                logger.debug("Skipping malformed triple item: %s — %s", item, exc)

        return triples
