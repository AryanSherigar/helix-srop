"""
POST /v1/sessions — create a session.
"""
import uuid
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import Session as SessionModel
from app.db.models import User as UserModel
from app.srop.state import SessionState

router = APIRouter(tags=["sessions"])


class CreateSessionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    plan_tier: Literal["free", "pro", "enterprise"] = "free"


class CreateSessionResponse(BaseModel):
    session_id: str


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    """
    Create a new session. Upsert the user if not seen before.
    Initialize SessionState and persist to DB.
    """
    session_id = str(uuid.uuid4())

    result = await db.execute(
        select(UserModel).where(UserModel.user_id == body.user_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        user = UserModel(user_id=body.user_id, plan_tier=body.plan_tier)
        db.add(user)
    elif user.plan_tier != body.plan_tier:
        user.plan_tier = body.plan_tier

    state = SessionState(
        user_id=body.user_id,
        plan_tier=body.plan_tier,
        last_agent=None,
        last_ticket_id=None,
        open_ticket_ids=[],
        turn_count=0,
    )
    session = SessionModel(
        session_id=session_id,
        user_id=body.user_id,
        state=state.to_db_dict(),
    )
    db.add(session)

    await db.commit()
    await db.refresh(session)

    return CreateSessionResponse(session_id=session.session_id)
