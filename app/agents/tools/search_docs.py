"""
search_docs tool — used by KnowledgeAgent.

Queries the vector store for relevant documentation chunks.
Returns chunk IDs, scores, and content so the agent can cite sources.

TODO for candidate: implement this tool.
Wire it to your chosen vector store (Chroma, LanceDB, FAISS, etc.).
"""
import asyncio
from dataclasses import dataclass

import chromadb
from google import genai

from app.settings import settings


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict  # e.g. {"product_area": "security", "source": "deploy-keys.md"}


async def search_docs(query: str, k: int = 5, product_area: str | None = None) -> list[DocChunk]:
    """
    Search the vector store for top-k relevant chunks.

    Args:
        query: natural language query from the user
        k: number of chunks to return
        product_area: optional metadata filter (e.g. "security", "ci-cd")

    Returns:
        List of DocChunk ordered by descending similarity score.
        Scores are rounded to 2 decimals to support citation formatting.

    Design considerations:
    - How do you embed the query? Same model as at ingest time.
    - Do you apply a score threshold to filter low-quality results?
    - How do you format chunks for the agent? Include chunk_id so agent can cite.
    """
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is required to search docs")

    query_embedding = await asyncio.to_thread(_embed_query, query)
    results = await asyncio.to_thread(_query_chroma, query_embedding, k, product_area)

    ids = (results.get("ids") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    chunks: list[DocChunk] = []
    for chunk_id, distance, document, metadata in zip(ids, distances, documents, metadatas):
        score = 1.0 - float(distance)
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        if score < 0.6:
            continue
        chunks.append(
            DocChunk(
                chunk_id=chunk_id,
                score=round(score, 2),
                content=document,
                metadata=metadata or {},
            )
        )

    return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)


def _embed_query(query: str) -> list[float]:
    client = genai.Client(api_key=settings.google_api_key)
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=[query],
            config={"task_type": "RETRIEVAL_QUERY"},
        )
        if hasattr(response, "embeddings") and response.embeddings:
            return list(response.embeddings[0].values)
        if hasattr(response, "embedding") and response.embedding:
            return list(response.embedding.values)
        raise ValueError("Unexpected embedding response shape")
    finally:
        client.close()


def _query_chroma(
    query_embedding: list[float],
    k: int,
    product_area: str | None,
) -> dict:
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = chroma_client.get_or_create_collection(
        name="helix_docs",
        metadata={"hnsw:space": "cosine"},
    )

    where = {"product_area": product_area} if product_area else None
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
    )
