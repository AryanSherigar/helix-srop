"""
Reranking evaluation helper.

Usage:
    python -m app.rag.rerank_eval
    python -m app.rag.rerank_eval --output docs/reranking-eval.md
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from app.agents.tools.search_docs import search_docs
from app.settings import settings

DEFAULT_QUERIES = [
    "How do I rotate a deploy key?",
    "How do I configure webhooks and verify signatures?",
    "What is the difference between Pro and Enterprise billing plans?",
    "How do I set up runners for Helix builds?",
    "How do I enable secret scanning for repos?",
]


def _format_results(title: str, results: list[dict[str, Any]]) -> str:
    lines: list[str] = [f"## {title}", ""]
    for item in results:
        lines.append(f"### Query: {item['query']}")
        lines.append("")
        for idx, chunk in enumerate(item["results"], start=1):
            source = chunk.get("source") or "unknown"
            score = chunk.get("score")
            score_text = f"{score:.2f}" if isinstance(score, (float, int)) else "n/a"
            lines.append(f"{idx}. {chunk['chunk_id']} (score={score_text}, source={source})")
        lines.append("")
    return "\n".join(lines).strip()


async def _run_queries(queries: list[str], k: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for query in queries:
        chunks = await search_docs(query, k=k)
        results.append(
            {
                "query": query,
                "results": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "score": chunk.score,
                        "source": (chunk.metadata or {}).get("source"),
                    }
                    for chunk in chunks
                ],
            }
        )
    return results


async def _evaluate(queries: list[str], k: int) -> str:
    original_rerank = settings.rerank_enabled
    try:
        settings.rerank_enabled = False
        baseline = await _run_queries(queries, k)

        settings.rerank_enabled = True
        reranked = await _run_queries(queries, k)
    finally:
        settings.rerank_enabled = original_rerank

    header = [
        "# Reranking Eval (LLM-as-judge)",
        "",
        f"Settings: rerank_candidate_k={settings.rerank_candidate_k}, top_k={k}",
        "",
    ]
    return "\n".join(header) + "\n" + _format_results("Before (Vector Only)", baseline) + "\n\n" + _format_results(
        "After (LLM Rerank)", reranked
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM reranking for 5 queries")
    parser.add_argument("--output", type=Path, default=None, help="Write markdown output to file")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    content = asyncio.run(_evaluate(DEFAULT_QUERIES, args.k))
    if args.output:
        args.output.write_text(content, encoding="utf-8")
        print(f"Wrote results to {args.output}")
    else:
        print(content)


if __name__ == "__main__":
    main()
