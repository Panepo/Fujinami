"""
Abstract base class for all graph extractors.

Any extractor (spaCy, LLM, hybrid) implements this interface
so ragService can swap methods without changing api.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from graph_engine.models import Triple


class BaseExtractor(ABC):
    """Extract a list of triples from a single text chunk."""

    @abstractmethod
    def extract(self, text: str, source_doc: str) -> list[Triple]:
        """
        Parameters
        ----------
        text:
            A single text chunk (plain string).
        source_doc:
            Document identifier — stored on every node and edge.

        Returns
        -------
        list[Triple]
            May be empty if no entities / relations found.
        """
        ...

    def extract_batch(self, chunks: list[str], source_doc: str) -> list[Triple]:
        """
        Extract triples from multiple chunks.

        Default implementation calls extract() sequentially.
        Override in subclasses that support true batch inference.
        """
        triples: list[Triple] = []
        for chunk in chunks:
            triples.extend(self.extract(chunk, source_doc))
        return triples
