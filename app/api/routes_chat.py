"""
POST /v1/chat/{session_id} — send a user message, get assistant reply.
"""
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentTrace, Message
from app.db.session import get_db
from app.srop import pipeline

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str
    routed_to: str   # which sub-agent handled this turn
    trace_id: str


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    accept: str | None = Header(default=None, alias="Accept"),
) -> ChatResponse | StreamingResponse:
    """
    Run one turn of the SROP pipeline.

    Error cases:
    - Session not found → 404
    - LLM timeout → 504
    """
    response = await _run_chat(session_id, body, db, idempotency_key)
    if _is_sse_request(accept):
        return StreamingResponse(
            _sse_stream(response),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return response


async def _run_chat(
    session_id: str,
    body: ChatRequest,
    db: AsyncSession,
    idempotency_key: str | None,
) -> ChatResponse:
    if idempotency_key:
        cached = await _get_cached_response(session_id, idempotency_key, db)
        if cached is not None:
            return cached

    try:
        result = await pipeline.run(
            session_id,
            body.message,
            db,
            idempotency_key=idempotency_key,
        )
    except IntegrityError:
        await db.rollback()
        cached = await (
            _get_cached_response(session_id, idempotency_key, db)
            if idempotency_key
            else None
        )
        if cached is not None:
            return cached
        raise

    return ChatResponse(
        reply=result.content,
        routed_to=result.routed_to,
        trace_id=result.trace_id,
    )


def _is_sse_request(accept: str | None) -> bool:
    if not accept:
        return False
    return "text/event-stream" in accept.lower()


async def _sse_stream(response: ChatResponse) -> AsyncGenerator[str, None]:
    payload = {
        "type": "final",
        "payload": {
            "reply": response.reply,
            "routed_to": response.routed_to,
            "trace_id": response.trace_id,
        },
    }
    yield _format_sse_data(payload)


def _format_sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _get_cached_response(
    session_id: str,
    idempotency_key: str,
    db: AsyncSession,
) -> ChatResponse | None:
    result = await db.execute(
        select(Message).where(
            Message.session_id == session_id,
            Message.idempotency_key == idempotency_key,
            Message.role == "assistant",
        )
    )
    cached_message = result.scalar_one_or_none()
    if cached_message is None:
        return None

    routed_to = "smalltalk"
    trace_id = cached_message.trace_id or ""
    if cached_message.trace_id:
        trace_result = await db.execute(
            select(AgentTrace).where(AgentTrace.trace_id == cached_message.trace_id)
        )
        trace = trace_result.scalar_one_or_none()
        if trace is not None:
            routed_to = trace.routed_to

    return ChatResponse(
        reply=cached_message.content,
        routed_to=routed_to,
        trace_id=trace_id,
    )
