"""
Integration tests — exercise the full SROP pipeline.
LLM mocked at the ADK boundary (not at the HTTP layer).
"""
import pytest

from app.api.routes_chat import _get_cached_response
from app.db.models import AgentTrace, Message, Session, User


REFUSAL_TEXT = "I can only help with Helix product and account questions."


@pytest.mark.asyncio
async def test_create_session(client):
    resp = await client.post("/v1/sessions", json={"user_id": "u_test_001"})
    assert resp.status_code == 200
    assert "session_id" in resp.json()
    assert "user_id" not in resp.json()


@pytest.mark.asyncio
async def test_knowledge_query_routes_correctly(client, mock_adk):
    """
    Core integration test.

    Sends a knowledge question, asserts:
    1. Response contains a reply
    2. routed_to == "knowledge"
    3. trace exists with retrieved chunk IDs
     4. Turn 2 in the same session has access to context from turn 1
         (state persistence — last_agent should reflect turn 1 routing)

    Implement after pipeline.run() and state persistence are working.
    The mock_adk fixture must patch at the ADK boundary, not at the HTTP layer.
    """
    # Create session
    sess = await client.post("/v1/sessions", json={"user_id": "u_test_002", "plan_tier": "pro"})
    session_id = sess.json()["session_id"]

    # Turn 1 — knowledge query
    r1 = await client.post(
        f"/v1/chat/{session_id}",
        json={"message": "How do I rotate a deploy key?"},
    )
    assert r1.status_code == 200
    assert r1.json()["routed_to"] == "knowledge"
    trace_id = r1.json()["trace_id"]

    # Trace must have chunk IDs
    trace = await client.get(f"/v1/traces/{trace_id}")
    assert trace.status_code == 200
    assert len(trace.json()["retrieved_chunk_ids"]) > 0

    # Turn 2 — follow-up in same session
    r2 = await client.post(
        f"/v1/chat/{session_id}",
        json={"message": "Which agent handled my last request?"},
    )
    assert r2.status_code == 200
    # Agent should know last_agent from state — not re-ask
    assert "knowledge" in r2.json()["reply"].lower()


@pytest.mark.asyncio
async def test_out_of_scope_refusal(client, mock_adk):
    sess = await client.post("/v1/sessions", json={"user_id": "u_test_005"})
    session_id = sess.json()["session_id"]

    resp = await client.post(
        f"/v1/chat/{session_id}",
        json={"message": "Write me a poem about clouds"},
    )
    assert resp.status_code == 200
    assert resp.json()["reply"] == REFUSAL_TEXT


@pytest.mark.asyncio
async def test_session_not_found_returns_404(client):
    resp = await client.post("/v1/chat/nonexistent-id", json={"message": "hello"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_idempotency_key_replays_response(client, mock_adk):
    sess = await client.post("/v1/sessions", json={"user_id": "u_test_003"})
    session_id = sess.json()["session_id"]

    headers = {"Idempotency-Key": "idem-test-001"}
    r1 = await client.post(
        f"/v1/chat/{session_id}",
        json={"message": "How do I rotate a deploy key?"},
        headers=headers,
    )
    assert r1.status_code == 200

    r2 = await client.post(
        f"/v1/chat/{session_id}",
        json={"message": "How do I rotate a deploy key?"},
        headers=headers,
    )
    assert r2.status_code == 200
    assert r2.json() == r1.json()
    assert mock_adk["count"] == 1


@pytest.mark.asyncio
async def test_get_cached_response_returns_payload(db):
    user_id = "u_test_004"
    session_id = "s_test_004"
    trace_id = "trace_test_004"

    db.add(User(user_id=user_id, plan_tier="free"))
    db.add(Session(session_id=session_id, user_id=user_id, state={}))
    db.add(
        AgentTrace(
            trace_id=trace_id,
            session_id=session_id,
            routed_to="knowledge",
            tool_calls=[],
            retrieved_chunk_ids=[],
            latency_ms=0,
        )
    )
    db.add(
        Message(
            message_id="msg_test_004",
            session_id=session_id,
            role="assistant",
            content="cached reply",
            idempotency_key="idem-test-004",
            trace_id=trace_id,
        )
    )
    await db.commit()

    cached = await _get_cached_response(session_id, "idem-test-004", db)
    assert cached is not None
    assert cached.reply == "cached reply"
    assert cached.routed_to == "knowledge"
    assert cached.trace_id == trace_id

    missing = await _get_cached_response(session_id, "idem-missing", db)
    assert missing is None
