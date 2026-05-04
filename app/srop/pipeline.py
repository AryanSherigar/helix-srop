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
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import structlog
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.agent_tool import AgentTool
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import HelixError, SessionNotFoundError, UpstreamTimeoutError
from app.agents.account import account_agent
from app.agents.knowledge import knowledge_agent
from app.agents.orchestrator import ROOT_INSTRUCTION
from app.db.models import AgentTrace, Message, Session as SessionModel
from app.settings import settings
from app.srop.state import SessionState


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

    result = await db.execute(
        select(SessionModel).where(SessionModel.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise SessionNotFoundError(f"Session {session_id} does not exist")

    structlog.contextvars.bind_contextvars(
        session_id=session_id,
        trace_id=trace_id,
    )
    if not session.state:
        log.error("session_state_missing")
        raise HelixError(f"Session state missing for session {session_id}")

    try:
        state = SessionState.from_db_dict(session.state)
    except ValidationError as exc:
        log.error("session_state_invalid", errors=exc.errors())
        raise HelixError(f"Invalid session state for session {session_id}") from exc

    structlog.contextvars.bind_contextvars(user_id=state.user_id)

    log.info("pipeline_started", user_message_len=len(user_message))
    started_at = time.perf_counter()

    instruction = _build_root_instruction(state)
    root_agent = _build_root_agent(instruction)

    try:
        content, routed_to, tool_calls, retrieved_chunk_ids = await asyncio.wait_for(
            _run_adk(root_agent, state.user_id, session_id, user_message),
            timeout=settings.llm_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        log.warning("pipeline_timeout", timeout_seconds=settings.llm_timeout_seconds)
        raise UpstreamTimeoutError(
            f"LLM did not respond within {settings.llm_timeout_seconds}s"
        ) from exc

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    routed_to = _normalize_routed_to(routed_to)

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
        raise

    log.info(
        "pipeline_completed",
        routed_to=routed_to,
        latency_ms=latency_ms,
        tool_calls=len(tool_calls),
    )

    return PipelineResult(content=content, routed_to=routed_to, trace_id=trace_id)


def _build_root_instruction(state: SessionState) -> str:
    return (
        f"{ROOT_INSTRUCTION}\n\n"
        "Current user context:\n"
        f"- user_id: {state.user_id}\n"
        f"- plan_tier: {state.plan_tier}\n"
        f"- last_agent: {state.last_agent}\n"
        f"- turn_count: {state.turn_count}\n"
    )


def _build_root_agent(instruction: str) -> LlmAgent:
    return LlmAgent(
        name="srop_root",
        model=settings.adk_model,
        instruction=instruction,
        tools=[AgentTool(agent=knowledge_agent), AgentTool(agent=account_agent)],
    )


async def _run_adk(
    agent: LlmAgent,
    user_id: str,
    session_id: str,
    user_message: str,
) -> tuple[str, str | None, list[dict[str, Any]], list[str]]:
    runner = InMemoryRunner(agent=agent)
    response = await runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message={"role": "user", "parts": [{"text": user_message}]},
    )

    routed_to: str | None = None
    tool_calls: list[dict[str, Any]] = []
    retrieved_chunk_ids: list[str] = []
    final_text = ""

    async for event in response:
        event_type = getattr(event, "type", None)

        if event_type == "tool_call":
            tool_calls.append(
                {
                    "tool_name": getattr(event, "tool_name", ""),
                    "args": _json_safe(getattr(event, "tool_args", {})),
                    "result": None,
                }
            )

        if event_type == "tool_result":
            tool_name = getattr(event, "tool_name", "")
            raw_result = _get_tool_result(event)
            result = _json_safe(raw_result)
            call = _find_pending_call(tool_calls, tool_name)
            if call is not None:
                call["result"] = result

            if tool_name == "search_docs":
                retrieved_chunk_ids.extend(_extract_chunk_ids(result))

        if hasattr(event, "is_final_response") and event.is_final_response():
            routed_to = getattr(event, "author", routed_to)
            content = getattr(event, "content", None)
            parts = getattr(content, "parts", []) if content is not None else []
            if parts:
                final_text = getattr(parts[0], "text", "") or ""
            else:
                final_text = ""

    return final_text, routed_to, tool_calls, retrieved_chunk_ids


def _get_tool_result(event: Any) -> Any:
    if hasattr(event, "result"):
        return event.result
    if hasattr(event, "tool_result"):
        return event.tool_result
    if hasattr(event, "content"):
        return event.content
    return None


def _find_pending_call(tool_calls: list[dict[str, Any]], tool_name: str) -> dict[str, Any] | None:
    for call in reversed(tool_calls):
        if call["tool_name"] == tool_name and call.get("result") is None:
            return call
    return None


def _extract_chunk_ids(result: Any) -> list[str]:
    chunk_ids: list[str] = []
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and item.get("chunk_id"):
                chunk_ids.append(str(item["chunk_id"]))
    elif isinstance(result, dict) and result.get("chunk_id"):
        chunk_ids.append(str(result["chunk_id"]))
    return chunk_ids


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_safe(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_routed_to(routed_to: str | None) -> str:
    if routed_to in {"knowledge", "account", "smalltalk"}:
        return routed_to
    if routed_to == "srop_root" or routed_to is None:
        return "smalltalk"
    return routed_to
