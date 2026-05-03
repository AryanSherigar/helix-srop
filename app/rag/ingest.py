"""
RAG ingest CLI.

Usage:
    python -m app.rag.ingest --path docs/
    python -m app.rag.ingest --path docs/ --chunk-size 512 --chunk-overlap 64

Reads markdown files, chunks them, embeds, and writes to the vector store.

TODO for candidate: implement chunking and embedding logic.
"""
import argparse
import asyncio
import hashlib
import re
from pathlib import Path

import chromadb
import yaml
from google import genai

from app.settings import settings


def _strip_frontmatter(text: str) -> str:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return text
    return text[match.end() :]


def _overlap_by_chars(sentences: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0:
        return []
    overlap: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        overlap.insert(0, sentence)
        total += len(sentence)
        if total >= overlap_chars:
            break
    return overlap


def _chunk_sentences(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if not sentence:
            continue
        if current_len + len(sentence) > max_chars and current:
            chunks.append(" ".join(current).strip())
            current = _overlap_by_chars(current, overlap_chars)
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c]


def chunk_markdown(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split markdown text into overlapping chunks.

    Design considerations:
    - Simple character splitting is fast but breaks mid-sentence.
    - Sentence-aware splitting is better for retrieval quality.
    - Heading-aware splitting (split on ## / ###) keeps sections coherent.
    - Overlap helps preserve context at chunk boundaries.

    Choose an approach and document why in the README.
    """
    body = _strip_frontmatter(text)
    sections = re.split(r"\n(?=#{2,3} )", body)
    chunks: list[str] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            chunks.extend(_chunk_sentences(section, max_chars=chunk_size, overlap_chars=overlap))

    return [c for c in chunks if c.strip()]


def extract_metadata(file_path: Path, text: str) -> dict:
    """
    Extract metadata from a markdown file's frontmatter.

    Expected frontmatter format:
        ---
        title: Deploy Keys
        product_area: security
        tags: [keys, secrets]
        ---

    Returns a dict suitable for vector store metadata filtering.
    """
    metadata: dict = {}
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if match:
        parsed = yaml.safe_load(match.group(1)) or {}
        if isinstance(parsed, dict):
            for key in ("title", "product_area", "tags"):
                if key in parsed:
                    metadata[key] = parsed[key]

    metadata["source"] = file_path.name
    return metadata


def _make_chunk_id(relative_path: str, chunk_index: int) -> str:
    raw = f"{relative_path}::{chunk_index}"
    return "chunk_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _extract_embeddings(response: object) -> list[list[float]]:
    if hasattr(response, "embeddings") and response.embeddings:
        return [list(embedding.values) for embedding in response.embeddings]
    if hasattr(response, "embedding") and response.embedding:
        return [list(response.embedding.values)]
    raise ValueError("Unexpected embedding response shape")


def _embed_texts(client: genai.Client, texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config={"task_type": "RETRIEVAL_DOCUMENT"},
    )
    return _extract_embeddings(response)


def _embed_in_batches(
    client: genai.Client,
    texts: list[str],
    batch_size: int = 20,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings.extend(_embed_texts(client, batch))
    return embeddings


async def ingest_directory(docs_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    """
    Walk docs_path, chunk and embed every .md file, upsert into vector store.

    Design considerations:
    - Generate a stable chunk_id (e.g. sha256(file + chunk_index)) for deduplication.
    - Run embeddings in batches to avoid rate limiting.
    - Print progress so the user can see what's happening.
    """
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is required to embed docs")

    genai_client = genai.Client(api_key=settings.google_api_key)
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = chroma_client.get_or_create_collection(
        name="helix_docs",
        metadata={"hnsw:space": "cosine"},
    )

    md_files = list(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in {docs_path}")

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        metadata = extract_metadata(file_path, text)
        chunks = chunk_markdown(text, chunk_size, chunk_overlap)
        print(f"  {file_path.name}: {len(chunks)} chunks")
        if not chunks:
            continue

        relative_path = file_path.relative_to(docs_path).as_posix()
        chunk_ids = [_make_chunk_id(relative_path, i) for i in range(len(chunks))]
        metadatas = [dict(metadata) for _ in chunks]
        embeddings = _embed_in_batches(genai_client, chunks, batch_size=20)

        collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

    genai_client.close()
    print("Ingest complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True, help="Directory containing .md files")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    asyncio.run(ingest_directory(args.path, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()
