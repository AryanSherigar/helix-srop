# Helix SROP — [Your Name]

## Setup

```bash
git clone <your-repo>
cd helix-srop
uv sync
cp .env.example .env  # fill in GOOGLE_API_KEY
uv run python -m app.rag.ingest --path docs/
uv run uvicorn app.main:app --reload
```

## Quick Test

```bash
SESSION=$(curl -s -X POST localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_demo", "plan_tier": "pro"}' | jq -r .session_id)

curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "How do I rotate a deploy key?"}' | jq .
```

## Architecture

```
[ASCII diagram here]
```

## Design Decisions

### State persistence (which pattern and why)
I used [Pattern 1/2/3 from the ADK guide] because...

### Chunking strategy
I used [heading-aware / sentence-aware / fixed-size] chunking because...

### Vector store choice
I chose [Chroma / LanceDB / FAISS] because...

## Known Limitations

- ...

## What I'd Do With More Time

- ...

## Time Spent

| Phase | Time |
|-------|------|
| Setup + DB + FastAPI boilerplate | |
| RAG ingest + search_docs | |
| ADK agents | |
| pipeline.py + state persistence | |
| Tests | |
| README | |
| **Total** | |

## Extensions Completed

- [ ] E1: Idempotency
- [ ] E2: Escalation agent
- [ ] E3: Streaming SSE
- [ ] E4: Reranking
- [ ] E5: Guardrails
- [ ] E6: Docker
- [ ] E7: Eval harness
