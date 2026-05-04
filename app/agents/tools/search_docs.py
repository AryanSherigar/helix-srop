"""
search_docs tool — used by KnowledgeAgent.

Queries the vector store for relevant documentation chunks.
Returns chunk IDs, scores, and content so the agent can cite sources.

TODO for candidate: implement this tool.
Wire it to your chosen vector store (Chroma, LanceDB, FAISS, etc.).
"""
import asyncio
import logging
import re
from dataclasses import dataclass

import chromadb
from google import genai

from app.settings import settings

logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 0.6
MAX_RERANK_CONTENT_CHARS = 800


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

    candidate_k = k
    if settings.rerank_enabled:
        candidate_k = max(k, settings.rerank_candidate_k)

    query_embedding = await asyncio.to_thread(_embed_query, query)
    results = await asyncio.to_thread(_query_chroma, query_embedding, candidate_k, product_area)

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
        if score < SCORE_THRESHOLD:
            continue
        chunks.append(
            DocChunk(
                chunk_id=chunk_id,
                score=round(score, 2),
                content=document,
                metadata=metadata or {},
            )
        )

    chunks = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
    if not chunks:
        return []

    if settings.rerank_enabled and len(chunks) > 1:
        return await _rerank_with_llm(query, chunks, top_k=k)

    return chunks[:k]


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


def _format_chunks_for_rerank(chunks: list[DocChunk]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        source = chunk.metadata.get("source") if chunk.metadata else None
        content = (chunk.content or "").strip()
        if len(content) > MAX_RERANK_CONTENT_CHARS:
            content = content[:MAX_RERANK_CONTENT_CHARS].rstrip() + "..."
        parts.append(
            f"[{chunk.chunk_id}] (score: {chunk.score:.2f}, source: {source})\n{content}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_text(response: object) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list) and candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None)
        if isinstance(parts, list):
            joined = "".join(
                [getattr(part, "text", "") for part in parts if getattr(part, "text", None)]
            ).strip()
            if joined:
                return joined

    return str(response)


def _call_rerank_model(prompt: str) -> str:
    client = genai.Client(api_key=settings.google_api_key)
    try:
        response = client.models.generate_content(
            model=settings.adk_model,
            contents=prompt,
            config={"temperature": 0},
        )
        return _extract_text(response).strip()
    finally:
        client.close()


def _parse_rerank_order(text: str, allowed_ids: set[str]) -> list[str]:
    if not text:
        return []
    ordered_ids: list[str] = []
    for match in re.findall(r"chunk_[A-Za-z0-9_-]+", text):
        if match in allowed_ids and match not in ordered_ids:
            ordered_ids.append(match)
    return ordered_ids


async def _rerank_with_llm(query: str, chunks: list[DocChunk], top_k: int) -> list[DocChunk]:
    prompt = (
        "Query: "
        + query
        + "\n\n"
        + "Rank these chunks by relevance (most relevant first). "
        + "Return only chunk IDs, comma-separated.\n\n"
        + "Chunks:\n"
        + _format_chunks_for_rerank(chunks)
    )

    try:
        text = await asyncio.wait_for(
            asyncio.to_thread(_call_rerank_model, prompt),
            timeout=settings.tool_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("rerank_timeout", extra={"timeout_seconds": settings.tool_timeout_seconds})
        return chunks[:top_k]
    except Exception as exc:
        logger.exception("rerank_failed", extra={"error": str(exc)})
        return chunks[:top_k]

    allowed_ids = {chunk.chunk_id for chunk in chunks}
    ordered_ids = _parse_rerank_order(text, allowed_ids)
    if not ordered_ids:
        logger.warning("rerank_parse_failed")
        return chunks[:top_k]

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    reranked: list[DocChunk] = []
    seen: set[str] = set()
    for chunk_id in ordered_ids:
        chunk = by_id.get(chunk_id)
        if chunk and chunk_id not in seen:
            reranked.append(chunk)
            seen.add(chunk_id)

    for chunk in chunks:
        if chunk.chunk_id not in seen:
            reranked.append(chunk)

    return reranked[:top_k]


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