"""
Unit tests for RAG retrieval.
Requires the vector store to be seeded first (run ingest.py on docs/).
"""
import pytest


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids(monkeypatch):
    """search_docs must return chunk IDs and scores in [0, 1]."""
    from app.agents.tools import search_docs as search_module
    from app.settings import settings

    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(search_module, "_embed_query", lambda query: [0.1, 0.2, 0.3])

    def fake_query_chroma(query_embedding, k, product_area):
        return {
            "ids": [["chunk_a", "chunk_b", "chunk_c"]],
            "distances": [[0.1, 0.2, 0.7]],
            "documents": [["doc a", "doc b", "doc c"]],
            "metadatas": [[{"source": "a"}, {"source": "b"}, {"source": "c"}]],
        }

    monkeypatch.setattr(search_module, "_query_chroma", fake_query_chroma)

    results = await search_module.search_docs("how to rotate a deploy key", k=3)
    assert len(results) > 0
    assert all(r.chunk_id for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_chunker_produces_non_empty_chunks():
    """Chunker must not produce empty strings."""
    from app.rag.ingest import chunk_markdown

    text = "# Header\n\nSome content.\n\n## Section 2\n\nMore content here."
    chunks = chunk_markdown(text, chunk_size=100, overlap=20)
    assert len(chunks) > 0
    assert all(c.strip() for c in chunks)
