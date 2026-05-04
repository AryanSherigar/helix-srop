# Helix SROP - Aryan Sherigar

Stateful RAG Orchestration Pipeline (SROP) for Helix support: one conversational API that routes between docs Q&A (RAG), account lookups, and escalations while persisting cross-turn context in SQLite.

## Setup

```bash
git clone <your-repo>
cd helix-srop
uv sync --extra dev
cp .env.example .env
# set GOOGLE_API_KEY in .env
uv run python -m app.rag.ingest --path docs/
uv run uvicorn app.main:app --reload
```

### Health check

```bash
curl -s localhost:8000/healthz | jq .
```

## API Quick Test

```bash
SESSION=$(curl -s -X POST localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_demo", "plan_tier": "pro"}' | jq -r .session_id)

curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I rotate a deploy key?"}' | jq .
```

Trace inspection:

```bash
TRACE_ID=<trace_id_from_chat_response>
curl -s localhost:8000/v1/traces/$TRACE_ID | jq .
```

## Idempotency

Use `Idempotency-Key` on `POST /v1/chat/{session_id}`.
Duplicate requests with the same key return the original response without re-running pipeline execution.

```bash
curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: idem-demo-001" \
  -d '{"message": "How do I rotate a deploy key?"}' | jq .
```

## Architecture

```text
POST /v1/chat/{session_id}
         |
         v
+----------------------------+
| SROP Pipeline              |
| 1) Load SessionState (DB)  |
| 2) Build ADK root agent    |
| 3) Run with timeout        |
| 4) Persist state/messages  |
| 5) Persist trace           |
+---------------+------------+
                |
        AgentTool routing
        /        |         \
 knowledge   account    escalation
    |           |           |
search_docs   build/status  create_ticket
    |
Chroma vector store + docs chunks
```

## Design Decisions

### State persistence pattern
I used **durable app-managed session state** (persisted JSON in `sessions.state`) and **re-injected that state every turn through root-agent system instruction**. This satisfies restart durability and keeps the schema explicit (`user_id`, `plan_tier`, `last_agent`, `last_ticket_id`, `open_ticket_ids`, `turn_count`).

Why this pattern:
- Survives process restarts (state is in SQLite, not memory).
- Small, deterministic state envelope per session.
- Easy to inspect/debug during review.

### Chunking strategy
I used a **hybrid heading-aware + sentence-aware chunker**:
- split markdown by `##`/`###` section boundaries first,
- then apply sentence-aware chunking with overlap for oversized sections.

Why: it preserves topical boundaries from docs while reducing mid-sentence fragmentation and improving retrieval quality.

### Stable chunk IDs
Chunk IDs are deterministic hashes of `relative_path::chunk_index`, so re-ingest upserts instead of duplicating vectors.

### Vector store choice
I used **Chroma (persistent local directory)** for simple local startup and straightforward metadata + similarity query support.

## Features Implemented

- Async FastAPI endpoints:
  - `POST /v1/sessions`
  - `POST /v1/chat/{session_id}`
  - `GET /v1/traces/{trace_id}`
  - `GET /healthz`
- ADK root orchestrator routes via **AgentTool** (knowledge/account/escalation).
- LLM timeout handling with mapped `UPSTREAM_TIMEOUT` error.
- Session-not-found and trace-not-found errors.
- Trace payload stores routed agent, tool calls, retrieved chunk IDs, latency.
- Optional SSE final-event response when `Accept: text/event-stream`.
- Idempotent chat replay using `Idempotency-Key`.

## Known Limitations

- `pytest-asyncio` is required for running tests; ensure dev dependencies are installed.
- End-to-end LLM behavior depends on valid provider credentials and model availability.
- Reranking quality evaluation exists, but not all extension items are production-hardened.

## What I’d Do With More Time

- Add production retry/backoff policy around model + tool failures.
- Expand refusal/guardrails test matrix and PII log redaction checks.
- Add Docker compose one-command smoke test.
- Add higher-volume concurrency and idempotency race-condition tests.

## Time Spent

| Phase | Time |
|---|---:|
| Setup + DB + FastAPI | 1h 20min |
| RAG ingest + retrieval | 1h 20min |
| ADK agent orchestration | 1h 30min |
| Pipeline/state/trace | 1h 40min |
| Tests + docs | 1h 10min |
| **Total** | **~7h 00min** |

## Extensions Completed

- [✓] E1: Idempotency
- [✓] E2: Escalation agent
- [✓] E3: Streaming SSE
- [✓] E4: Reranking (see `docs/reranking-eval.md`)
- [✓] E5: Guardrails refusal behavior
- [ ] E6: Docker
- [ ] E7: Eval harness


## Model used during testing

Everything in this assignment was tested using **Gemini 3 Flash**.
