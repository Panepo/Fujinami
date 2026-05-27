"""
Token-based text chunker for graph extraction.

Splits a plain-text string into overlapping chunks so each chunk
fits inside an LLM context window without losing cross-sentence context.
"""
from __future__ import annotations

import re

_DEFAULT_CHUNK_SIZE = 1000  # tokens (words approximation)
_DEFAULT_OVERLAP = 200       # tokens overlap between consecutive chunks


def _tokenize(text: str) -> list[str]:
    """Split text into word-level tokens (simple whitespace split)."""
    return text.split()


def chunk_text(
    text: str,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping token windows.

    Parameters
    ----------
    text:
        Plain text to split.
    chunk_size:
        Approximate number of words per chunk.
    overlap:
        Number of words to repeat at the start of each subsequent chunk
        so entities spanning a boundary are not lost.

    Returns
    -------
    list[str]
        List of text chunks. Returns ``[text]`` unchanged if text is
        shorter than *chunk_size*.
    """
    tokens = _tokenize(text.strip())
    if not tokens:
        return []
    if len(tokens) <= chunk_size:
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        if end == len(tokens):
            break
        start = end - overlap

    return chunks


def chunk_sentences(
    text: str,
    max_sentences: int = 10,
    overlap_sentences: int = 2,
) -> list[str]:
    """
    Alternative chunker: split by sentences rather than tokens.

    Useful when sentence boundaries matter more than token count
    (e.g. dependency parsing in spaCy works best on complete sentences).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= max_sentences:
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        end = min(start + max_sentences, len(sentences))
        chunks.append(" ".join(sentences[start:end]))
        if end == len(sentences):
            break
        start = end - overlap_sentences

    return chunks
