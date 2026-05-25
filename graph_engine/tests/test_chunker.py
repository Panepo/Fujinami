"""
Tests for chunker behaviour.

Rules derive from:
  - graph-design-spec.md §6 — "Graph team owns chunking"
  - graph-generation-analysis.md §7 — chunk_size=1200 tokens, overlap=100 tokens
  - spec §7 pipeline: chunking is the first stage before LLM extraction

Tests verify observable splitting behaviour only.
No implementation code was read.
Public API: chunk_text(text, chunk_size, overlap) -> list[str]
"""

import pytest
from graph_engine.chunker import chunk_text


class TestChunkText:

    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size → exactly 1 chunk."""
        chunks = chunk_text("Hello world. This is a short document.", chunk_size=200, overlap=20)
        assert len(chunks) == 1

    def test_single_chunk_contains_original_text(self):
        text = "Hello world. This is a short document."
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        assert text.strip() in chunks[0] or chunks[0].strip() in text

    def test_long_text_returns_multiple_chunks(self):
        """Text longer than chunk_size → more than 1 chunk."""
        sentence = "The sensor module communicates with the central processing unit via RS-485 protocol. "
        long_text = sentence * 30  # ~900 words, well above chunk_size=100
        chunks = chunk_text(long_text, chunk_size=100, overlap=10)
        assert len(chunks) > 1

    def test_empty_string_returns_empty_list(self):
        chunks = chunk_text("", chunk_size=200, overlap=20)
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self):
        chunks = chunk_text("   \n\t  ", chunk_size=200, overlap=20)
        assert chunks == []

    def test_overlap_words_appear_in_consecutive_chunks(self):
        """
        With overlap > 0, the tail of chunk[N] and the head of chunk[N+1]
        must share at least some content (spec §7 overlap).
        """
        sentence = "The sensor reads temperature and sends data to the controller unit. "
        long_text = sentence * 20
        chunks = chunk_text(long_text, chunk_size=50, overlap=15)

        assert len(chunks) >= 2, "Need at least 2 chunks to check overlap"

        for i in range(len(chunks) - 1):
            tail_words = set(chunks[i].split()[-10:])
            head_words = set(chunks[i + 1].split()[:10])
            overlap_found = bool(tail_words & head_words)
            assert overlap_found, (
                f"No overlap found between chunk {i} and {i+1}.\n"
                f"  Tail: {chunks[i][-80:]!r}\n"
                f"  Head: {chunks[i+1][:80]!r}"
            )

    def test_all_text_covered_across_chunks(self):
        """Every word in the original text must appear in at least one chunk."""
        sentence = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet. "
        long_text = sentence * 15
        chunks = chunk_text(long_text, chunk_size=50, overlap=10)

        original_words = set(long_text.split())
        covered_words = set(w for chunk in chunks for w in chunk.split())
        missing = original_words - covered_words
        assert not missing, f"Words not covered by any chunk: {missing}"

    def test_chunk_size_respected(self):
        """Each chunk must not exceed chunk_size tokens by a large margin."""
        sentence = "sensor data protocol interface component standard version vendor. "
        long_text = sentence * 40
        chunk_size = 50
        chunks = chunk_text(long_text, chunk_size=chunk_size, overlap=5)

        for i, chunk in enumerate(chunks):
            word_count = len(chunk.split())
            assert word_count <= chunk_size * 2, (
                f"Chunk {i} has {word_count} words, far exceeds chunk_size={chunk_size}"
            )
