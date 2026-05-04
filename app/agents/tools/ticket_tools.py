"""
Ticket tools — used by EscalationAgent.
"""
import uuid

import structlog

from app.db.models import Ticket
from app.db.session import AsyncSessionLocal


_ALLOWED_PRIORITIES = {"critical", "high", "normal", "low"}


def _normalize_priority(priority: str | None) -> str:
    if not priority:
        return "normal"
    lowered = priority.strip().lower()
    return lowered if lowered in _ALLOWED_PRIORITIES else "normal"


def _get_session_id_from_context() -> str | None:
    context = structlog.contextvars.get_contextvars()
    session_id = context.get("session_id")
    return str(session_id) if session_id else None


async def create_ticket(user_id: str, summary: str, priority: str = "normal") -> dict[str, str]:
    """
    Create a support ticket and return its ID.

    Args:
        user_id: user opening the ticket
        summary: short description of the issue
        priority: critical | high | normal | low

    Returns:
        Dict with the new ticket_id.
    """
    log = structlog.get_logger()
    cleaned_summary = summary.strip() if summary else ""
    if not cleaned_summary:
        raise ValueError("summary is required")

    normalized_priority = _normalize_priority(priority)
    session_id = _get_session_id_from_context()
    if session_id is None:
        raise ValueError("session_id missing from context")

    ticket_id = f"tkt_{uuid.uuid4()}"

    async with AsyncSessionLocal() as db:
        db.add(
            Ticket(
                ticket_id=ticket_id,
                user_id=user_id,
                session_id=session_id,
                summary=cleaned_summary,
                priority=normalized_priority,
                status="open",
            )
        )
        try:
            await db.commit()
        except Exception:
            await db.rollback()
            log.exception("ticket_create_failed", ticket_id=ticket_id)
            raise

    log.info("ticket_created", ticket_id=ticket_id, priority=normalized_priority)
    return {"ticket_id": ticket_id}
