"""
SROP entrypoint — called by the message route.

This is the core of the assignment. It ties together:
  - Loading session state from DB
  - Running the ADK orchestrator with that state as context
  - Extracting routing decision and tool calls from ADK events
  - Recording the trace
  - Persisting updated session state to DB

The route calls: result = await pipeline.run(session_id, user_message, db)
It receives: PipelineResult(content, routed_to, trace_id)

Design questions you need to answer:
  1. How do you inject SessionState into the ADK agent so it knows the user's context?
     (system prompt injection vs ADK session state vs re-hydrating from message history)
  2. How do you determine WHICH sub-agent handled the turn from ADK's event stream?
  3. How do you capture tool calls (name, args, result) for the trace?
  4. What is your timeout strategy? (see settings.llm_timeout_seconds)
  5. If the DB write for state fails after the LLM responds, what do you do?

See docs/google-adk-guide.md for ADK event stream patterns.
"""
import asyncio
import time
import uuid
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
from typing import Any

import structlog
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import HelixError, SessionNotFoundError, UpstreamTimeoutError
from app.agents.account import account_agent
from app.agents.escalation import build_escalation_agent
from app.agents.knowledge import knowledge_agent
from app.agents.orchestrator import ROOT_INSTRUCTION
from app.db.models import AgentTrace, Message, Session as SessionModel
from app.settings import settings
from app.srop.state import SessionState


APP_NAME = "srop"
KNOWN_SUB_AGENTS = {"knowledge", "account", "escalation"}


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


async def run(
    session_id: str,
    user_message: str,
    db: AsyncSession,
    idempotency_key: str | None = None,
) -> PipelineResult:
    trace_id = str(uuid.uuid4())
    log = structlog.get_logger()

    # ---- Idempotency: if we've already produced an assistant reply for this key,
    # short-circuit and return it instead of re-running the LLM.
    if idempotency_key:
        existing = await db.execute(
            select(Message).where(
                Message.session_id == session_id,
                Message.idempotency_key == idempotency_key,
                Message.role == "assistant",
            )
        )
        prior = existing.scalar_one_or_none()
        if prior is not None:
            trace_row = await db.execute(
                select(AgentTrace).where(AgentTrace.trace_id == prior.trace_id)
            )
            trace = trace_row.scalar_one_or_none()
            return PipelineResult(
                content=prior.content,
                routed_to=trace.routed_to if trace else "smalltalk",
                trace_id=prior.trace_id,
            )

    # ---- Load (and lock) session row to prevent concurrent-turn races.
    result = await db.execute(
        select(SessionModel)
        .where(SessionModel.session_id == session_id)
        .with_for_update()
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise SessionNotFoundError(f"Session {session_id} does not exist")

    structlog.contextvars.bind_contextvars(session_id=session_id, trace_id=trace_id)
    try:
        if not session.state:
            log.error("session_state_missing")
            raise HelixError(f"Session state missing for session {session_id}")

        try:
            state = SessionState.from_db_dict(session.state)
        except ValidationError as exc:
            log.error("session_state_invalid", errors=exc.errors())
            raise HelixError(
                f"Invalid session state for session {session_id}"
            ) from exc

        structlog.contextvars.bind_contextvars(user_id=state.user_id)
        log.info("pipeline_started", user_message_len=len(user_message))
        started_at = time.perf_counter()

        instruction = _build_root_instruction(state)
        root_agent = _build_root_agent(instruction, state)

        try:
            content, routed_to, tool_calls, retrieved_chunk_ids = await asyncio.wait_for(
                _run_adk(root_agent, state.user_id, session_id, user_message),
                timeout=settings.llm_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            log.warning(
                "pipeline_timeout", timeout_seconds=settings.llm_timeout_seconds
            )
            raise UpstreamTimeoutError(
                f"LLM did not respond within {settings.llm_timeout_seconds}s"
            ) from exc

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        routed_to = _normalize_routed_to(routed_to, tool_calls)

        # ---- Mutate state only after a successful LLM call.
        _update_ticket_state(state, tool_calls)
        state.last_agent = routed_to
        state.turn_count += 1
        session.state = state.to_db_dict()

        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())

        db.add(
            Message(
                message_id=user_message_id,
                session_id=session_id,
                role="user",
                content=user_message,
                trace_id=trace_id,
            )
        )
        db.add(
            Message(
                message_id=assistant_message_id,
                session_id=session_id,
                role="assistant",
                content=content,
                trace_id=trace_id,
                idempotency_key=idempotency_key,
            )
        )
        db.add(
            AgentTrace(
                trace_id=trace_id,
                session_id=session_id,
                routed_to=routed_to,
                tool_calls=tool_calls,
                retrieved_chunk_ids=retrieved_chunk_ids,
                latency_ms=latency_ms,
            )
        )

        try:
            await db.commit()
        except Exception:
            await db.rollback()
            log.exception("pipeline_db_commit_failed")
            # The LLM call succeeded but persistence failed. We surface the
            # error so the caller can decide whether to retry; the
            # idempotency_key (if any) ensures a retry won't double-spend.
            raise

        log.info(
            "pipeline_completed",
            routed_to=routed_to,
            latency_ms=latency_ms,
            tool_calls=len(tool_calls),
        )
        return PipelineResult(
            content=content, routed_to=routed_to, trace_id=trace_id
        )
    finally:
        structlog.contextvars.clear_contextvars()


def _build_root_instruction(state: SessionState) -> str:
    return (
        f"{ROOT_INSTRUCTION}\n\n"
        "Current user context:\n"
        f"- user_id: {state.user_id}\n"
        f"- plan_tier: {state.plan_tier}\n"
        f"- last_agent: {state.last_agent}\n"
        f"- last_ticket_id: {state.last_ticket_id}\n"
        f"- open_ticket_ids: {state.open_ticket_ids}\n"
        f"- turn_count: {state.turn_count}\n"
    )


def _build_root_agent(instruction: str, state: SessionState) -> LlmAgent:
    return LlmAgent(
        name="srop_root",
        model=settings.adk_model,
        instruction=instruction,
        tools=[
            AgentTool(agent=knowledge_agent),
            AgentTool(agent=account_agent),
            AgentTool(agent=build_escalation_agent(state)),
        ],
    )


async def _run_adk(
    agent: LlmAgent,
    user_id: str,
    session_id: str,
    user_message: str,
) -> tuple[str, str | None, list[dict[str, Any]], list[str]]:
    """Invoke the ADK runner and harvest events for tracing."""
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)

    # ADK requires the session to exist before run_async; create it fresh per
    # turn since we re-inject our durable state via the system prompt.
    await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    new_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_message)],
    )

    routed_to: str | None = None
    # Map ADK function-call id -> our trace dict, so responses pair reliably
    # even when the same tool is called multiple times.
    pending_by_id: dict[str, dict[str, Any]] = {}
    tool_calls: list[dict[str, Any]] = []
    retrieved_chunk_ids: list[str] = []
    final_text_parts: list[str] = []

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message,
        ):
            # --- Tool calls (function calls issued by the LLM) ---
            for fc in (event.get_function_calls() or []):
                args = dict(fc.args) if fc.args else {}
                entry = {
                    "tool_name": fc.name,
                    "args": _json_safe(args),
                    "result": None,
                }
                tool_calls.append(entry)
                if fc.id:
                    pending_by_id[fc.id] = entry

            # --- Tool responses (results returned to the LLM) ---
            for fr in (event.get_function_responses() or []):
                result = _json_safe(fr.response)
                entry = pending_by_id.pop(fr.id, None) if fr.id else None
                if entry is None:
                    entry = _find_pending_call(tool_calls, fr.name)
                if entry is not None:
                    entry["result"] = result

                if fr.name == "search_docs":
                    retrieved_chunk_ids.extend(_extract_chunk_ids(result))

            # --- Final response text ---
            if event.is_final_response():
                routed_to = event.author or routed_to
                content = getattr(event, "content", None)
                parts = getattr(content, "parts", None) or []
                text_chunks = [
                    p.text for p in parts if getattr(p, "text", None)
                ]
                if text_chunks:
                    final_text_parts.extend(text_chunks)
    finally:
        # Best-effort cleanup of the in-memory session for this turn.
        close = getattr(runner, "close", None)
        if close is not None:
            try:
                maybe_coro = close()
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro
            except Exception:
                pass

    return "".join(final_text_parts), routed_to, tool_calls, retrieved_chunk_ids


def _find_pending_call(
    tool_calls: list[dict[str, Any]], tool_name: str
) -> dict[str, Any] | None:
    for call in reversed(tool_calls):
        if call["tool_name"] == tool_name and call.get("result") is None:
            return call
    return None


def _extract_chunk_ids(result: Any) -> list[str]:
    chunk_ids: list[str] = []
    if isinstance(result, dict):
        # search_docs may return {"results": [...]} or a single chunk dict.
        if isinstance(result.get("results"), list):
            for item in result["results"]:
                if isinstance(item, dict) and item.get("chunk_id"):
                    chunk_ids.append(str(item["chunk_id"]))
        elif result.get("chunk_id"):
            chunk_ids.append(str(result["chunk_id"]))
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and item.get("chunk_id"):
                chunk_ids.append(str(item["chunk_id"]))
    return chunk_ids


def _json_safe(value: Any) -> Any:
    """Recursively coerce ADK/Pydantic/dataclass objects to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Pydantic v2 model
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump())
    # Pydantic v1 fallback
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    # Dataclass instance (not the class itself)
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _json_safe(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _normalize_routed_to(
    routed_to: str | None, tool_calls: list[dict[str, Any]]
) -> str:
    """Determine which sub-agent handled the turn.

    Because sub-agents are wrapped as AgentTools, the final-response author is
    always the root agent. Infer routing from the AgentTool that was invoked.
    The *last* sub-agent tool call wins (most recent decision).
    """
    for call in reversed(tool_calls):
        name = (call.get("tool_name") or "").lower()
        # AgentTool exposes the wrapped agent's name as the tool name.
        if name in {"knowledge_agent", "knowledge"}:
            return "knowledge"
        if name in {"account_agent", "account"}:
            return "account"
        if name in {"escalation_agent", "escalation"}:
            return "escalation"

    if routed_to in KNOWN_SUB_AGENTS:
        return routed_to
    if routed_to in (None, "srop_root"):
        return "smalltalk"
    # Unknown author — log upstream, fall back to smalltalk.
    return "smalltalk"


def _update_ticket_state(state: SessionState, tool_calls: list[dict[str, Any]]) -> None:
    ticket_ids = _extract_ticket_ids(tool_calls)
    for ticket_id in ticket_ids:
        state.last_ticket_id = ticket_id
        if ticket_id not in state.open_ticket_ids:
            state.open_ticket_ids.append(ticket_id)


def _extract_ticket_ids(tool_calls: list[dict[str, Any]]) -> list[str]:
    ticket_ids: list[str] = []
    for call in tool_calls:
        name = (call.get("tool_name") or "").lower()
        if name != "create_ticket":
            continue
        result = call.get("result")
        if isinstance(result, dict) and result.get("ticket_id"):
            ticket_ids.append(str(result["ticket_id"]))
        elif isinstance(result, str) and result:
            ticket_ids.append(result)
    return ticket_ids