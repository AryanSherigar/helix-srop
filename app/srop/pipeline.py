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
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


async def run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
    trace_id = str(uuid.uuid4())
    # Your implementation goes here.
    ...
