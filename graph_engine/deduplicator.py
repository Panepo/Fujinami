"""
Node deduplication for the graph engine.

Two nodes are considered duplicates when they share the same
``name`` (case-insensitive, stripped) and ``type``.
Duplicate nodes are merged: their ``source_doc`` values are unioned
and their ``specs`` dicts are merged (first value wins on key conflict).
"""
from __future__ import annotations

from collections import defaultdict

from graph_engine.models import Edge, Node, Triple


def _node_key(node: Node) -> str:
    return f"{node.name.strip().lower()}::{node.type.strip().lower()}"


def deduplicate_triples(triples: list[Triple]) -> list[Triple]:
    """
    Merge duplicate nodes across a list of triples, then return
    a deduplicated triple list with stable node IDs.

    Steps
    -----
    1. Collect all nodes keyed by ``(name.lower(), type.lower())``.
    2. For each duplicate group, elect a canonical node (first seen,
       union source_docs, merge specs).
    3. Rewrite every triple's subject / object to the canonical node.
    4. Deduplicate edges: keep the highest-weight triple when
       (subject_id, predicate, object_id) repeats.
    """
    if not triples:
        return []

    # --- Step 1 & 2: build canonical node map ---
    canonical: dict[str, Node] = {}

    for triple in triples:
        for node in (triple.subject, triple.object):
            key = _node_key(node)
            if key not in canonical:
                canonical[key] = node.model_copy(deep=True)
                canonical[key].source_doc = node.source_doc
            else:
                canon = canonical[key]
                # Union source docs (store as comma-separated if different)
                existing_docs = set(canon.source_doc.split(","))
                existing_docs.add(node.source_doc)
                canon.source_doc = ",".join(sorted(existing_docs))
                # Merge specs (first value wins)
                for k, v in node.specs.items():
                    canon.specs.setdefault(k, v)

    # --- Step 3 & 4: rewrite and deduplicate triples ---
    seen_edges: dict[tuple, Triple] = {}

    for triple in triples:
        subj_key = _node_key(triple.subject)
        obj_key = _node_key(triple.object)

        new_triple = Triple(
            subject=canonical[subj_key],
            predicate=triple.predicate,
            object=canonical[obj_key],
            weight=triple.weight,
            source_doc=triple.source_doc,
            method=triple.method,
        )

        edge_key = (canonical[subj_key].id, triple.predicate, canonical[obj_key].id)
        if edge_key not in seen_edges or triple.weight > seen_edges[edge_key].weight:
            seen_edges[edge_key] = new_triple

    return list(seen_edges.values())
