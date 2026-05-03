"""
Unit tests for RAG retrieval.
Requires the vector store to be seeded first (run ingest.py on docs/).
"""
import pytest


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids():
    """search_docs must return chunk IDs and scores in [0, 1]."""
    # from app.agents.tools.search_docs import search_docs
    # results = await search_docs("how to rotate a deploy key", k=3)
    # assert len(results) > 0
    # assert all(r.chunk_id for r in results)
    # assert all(0.0 <= r.score <= 1.0 for r in results)
    pytest.skip("Implement after search_docs and ingest are working")


def test_chunker_produces_non_empty_chunks():
    """Chunker must not produce empty strings."""
    # from app.rag.ingest import chunk_markdown
    # text = "# Header\n\nSome content.\n\n## Section 2\n\nMore content here."
    # chunks = chunk_markdown(text, chunk_size=100, overlap=20)
    # assert len(chunks) > 0
    # assert all(c.strip() for c in chunks)
    pytest.skip("Implement after chunk_markdown is built")
