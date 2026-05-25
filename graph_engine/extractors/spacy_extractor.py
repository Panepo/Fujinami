"""
SpacyExtractor — NER + dependency parsing for fast graph extraction.

Uses spaCy's transformer model to:
  1. Identify named entities (NER)
  2. Traverse dependency tree to find subject-verb-object triples
  3. Map spaCy entity labels → closed entity type list (graph-design-spec §3)
  4. Map dependency-derived verbs → closed relation set (graph-design-spec §4)

Speed: ~50–200ms per chunk on CPU, ~10–30ms on GPU.
No API calls — runs fully local.

Trade-off: relation quality is lower than LLM; works best on
clear subject-verb-object sentences. Use HybridExtractor for better
relation labeling at minimal extra cost.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from graph_engine.base import BaseExtractor
from graph_engine.models import Node, Triple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping: spaCy entity label → closed entity type
# ---------------------------------------------------------------------------

_SPACY_TO_TYPE: dict[str, str] = {
    "PERSON": "Person",
    "PER": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "FAC": "Location",
    "DATE": "Date",
    "TIME": "Date",
    "EVENT": "Event",
    "PRODUCT": "Product",
    "WORK_OF_ART": "Concept",
    "LAW": "Standard",
    "LANGUAGE": "Software",
    "NORP": "Organization",
    # Technical / device labels — spaCy won't produce these natively
    # but a fine-tuned model would:
    "DEVICE": "Device",
    "COMPONENT": "Component",
    "PROTOCOL": "Protocol",
    "SOFTWARE": "Software",
    "STANDARD": "Standard",
    "ERROR": "Error",
    "VERSION": "Version",
    "VENDOR": "Vendor",
    "INTERFACE": "Interface",
    "CONFIGURATION": "Configuration",
    "SPECIFICATION": "Specification",
}

# ---------------------------------------------------------------------------
# Mapping: dependency verb root → closed relation type
# Verbs are lemmatized before lookup.
# ---------------------------------------------------------------------------

_VERB_TO_RELATION: dict[str, str] = {
    # is_a
    "be": "is_a",
    "represent": "is_a",
    "define": "is_a",
    "describe": "is_a",
    "classify": "is_a",
    # part_of
    "contain": "part_of",
    "include": "part_of",
    "comprise": "part_of",
    "consist": "part_of",
    "embed": "part_of",
    "integrate": "part_of",
    # causes
    "cause": "causes",
    "trigger": "causes",
    "generate": "causes",
    "produce": "causes",
    "lead": "causes",
    "result": "causes",
    "create": "causes",
    # governs
    "govern": "governs",
    "control": "governs",
    "manage": "governs",
    "regulate": "governs",
    "handle": "governs",
    "operate": "governs",
    "oversee": "governs",
    # supports_protocol
    "support": "supports_protocol",
    "implement": "supports_protocol",
    "use": "supports_protocol",
    "utilize": "supports_protocol",
    "communicate": "supports_protocol",
    # certified_by
    "certify": "certified_by",
    "comply": "certified_by",
    "conform": "certified_by",
    "meet": "certified_by",
    "satisfy": "certified_by",
    # opposes
    "oppose": "opposes",
    "conflict": "opposes",
    "contradict": "opposes",
    "prevent": "opposes",
    # related_to (fallback)
    "connect": "related_to",
    "link": "related_to",
    "associate": "related_to",
    "relate": "related_to",
    "measure": "related_to",
    "detect": "related_to",
    "monitor": "related_to",
    "transmit": "related_to",
    "receive": "related_to",
    "send": "related_to",
    "convert": "related_to",
    "provide": "related_to",
}

_DEFAULT_RELATION = "related_to"

# Labels that produce noise on markdown/table/prose content — skip these
_SKIP_LABELS: frozenset[str] = frozenset({
    "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT",
})


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Load spaCy model once and cache it.

    Prefers a locally saved copy under ``<project_root>/models/<model_name>``
    so the package does not need to be installed in the environment.
    """
    import spacy  # noqa: PLC0415

    _local = Path(__file__).resolve().parents[2] / "models" / model_name
    load_target: str | Path = _local if _local.exists() else model_name
    logger.info("Loading spaCy model from: %s", load_target)
    return spacy.load(load_target)


class SpacyExtractor(BaseExtractor):
    """
    Extract triples using spaCy NER + dependency parsing.

    Parameters
    ----------
    model_name:
        spaCy model to load. ``en_core_web_trf`` for best accuracy,
        ``en_core_web_sm`` for fastest speed.
    min_entity_length:
        Skip entity spans shorter than this (filters out noise like "a", "it").
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        min_entity_length: int = 3,
    ) -> None:
        self._model_name = model_name
        self._min_entity_length = min_entity_length
        self._nlp = _load_model(model_name)

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
        # Mostly punctuation / symbols / numbers
        alpha_chars = sum(1 for c in stripped if c.isalpha())
        if alpha_chars < 2:
            return True
        # Pure numbers (e.g. "120", "84,148")
        if stripped.replace(",", "").replace(".", "").replace("-", "").isdigit():
            return True
        return False

    def extract(self, text: str, source_doc: str) -> list[Triple]:
        if not text.strip():
            return []

        doc = self._nlp(text)
        entity_map = self._build_entity_map(doc)
        return self._extract_triples(doc, entity_map, source_doc)

    def _build_entity_map(self, doc) -> dict:
        """Map token index → Node for NER spans + noun chunks."""
        entity_map: dict[int, Node] = {}

        # 1. Named entities (highest priority — overwrites noun chunks)
        ner_token_ids: set[int] = set()
        for ent in doc.ents:
            if ent.label_ in _SKIP_LABELS:
                continue
            name = ent.text.strip()
            if len(name) < self._min_entity_length:
                continue
            if self._is_garbage_entity(name):
                continue
            ent_type = _SPACY_TO_TYPE.get(ent.label_, "Other")
            node = Node(name=name, type=ent_type, source_doc="")
            for token in ent:
                entity_map[token.i] = node
                ner_token_ids.add(token.i)

        # 2. Noun chunks as "Concept" entities (fill gaps NER missed)
        for chunk in doc.noun_chunks:
            # Skip if head already covered by NER
            if chunk.root.i in ner_token_ids:
                continue
            name = chunk.text.strip()
            if len(name) < self._min_entity_length:
                continue
            if self._is_garbage_entity(name):
                continue
            # Skip determiners/pronouns as sole content
            if chunk.root.pos_ in ("PRON", "DET"):
                continue
            node = Node(name=name, type="Concept", source_doc="")
            for token in chunk:
                if token.i not in ner_token_ids:
                    entity_map[token.i] = node

        return entity_map

    def _extract_triples(self, doc, entity_map: dict, source_doc: str) -> list[Triple]:
        triples: list[Triple] = []

        for token in doc:
            # Only process verbal roots / predicates
            if token.pos_ not in ("VERB", "AUX"):
                continue

            subject_node = self._find_dep(token, entity_map, {"nsubj", "nsubjpass"})
            object_node = self._find_dep(token, entity_map, {"dobj", "pobj", "attr", "nmod"})

            if subject_node is None or object_node is None:
                continue
            if subject_node.id == object_node.id:
                continue

            lemma = token.lemma_.lower()
            relation = _VERB_TO_RELATION.get(lemma, _DEFAULT_RELATION)

            triples.append(
                Triple(
                    subject=subject_node.model_copy(update={"source_doc": source_doc}),
                    predicate=relation,
                    object=object_node.model_copy(update={"source_doc": source_doc}),
                    weight=0.7,  # spaCy triples get a fixed lower weight than LLM
                    source_doc=source_doc,
                )
            )

        return triples

    def _find_dep(self, token, entity_map: dict, dep_labels: set) -> Node | None:
        """Return the first child token with a matching dependency label that is an entity."""
        for child in token.children:
            if child.dep_ not in dep_labels:
                continue
            # Direct hit
            if child.i in entity_map:
                return entity_map[child.i]
            # Walk one level deeper (e.g. compound noun head)
            for grandchild in child.children:
                if grandchild.i in entity_map:
                    return entity_map[grandchild.i]
        return None
