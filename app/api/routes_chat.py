"""
POST /v1/chat/{session_id} — send a user message, get assistant reply.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.srop import pipeline

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    reply: str
    routed_to: str   # which sub-agent handled this turn
    trace_id: str


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Run one turn of the SROP pipeline.

    Error cases:
    - Session not found → 404
    - LLM timeout → 504
    """
    result = await pipeline.run(session_id, body.content, db)
    return ChatResponse(reply=result.content, routed_to=result.routed_to, trace_id=result.trace_id)
