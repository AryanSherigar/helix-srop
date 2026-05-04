"""
Test fixtures.

Key fixtures:
- `client`: async test client with in-memory SQLite DB
- `mock_adk`: patches the ADK root agent so tests don't hit the real LLM
- `seeded_db`: DB with a test user and session pre-created
"""
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base
from app.db.session import get_db
from app.main import app


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
REFUSAL_TEXT = "I can only help with Helix product and account questions."

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client(db):
    """Async test client with DB overridden to in-memory SQLite."""
    app.dependency_overrides[get_db] = lambda: db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_adk(monkeypatch):
    """
    Patch the ADK pipeline so tests don't call the real LLM.

    TODO for candidate: patch at the ADK boundary (not at the HTTP layer).
    The mock should:
    1. Accept a user message
    2. Return a canned response with a specified routed_to value
    3. Allow tests to assert which sub-agent was called

    Example:
        def mock_run(session_id, message, db):
            if "rotate" in message.lower():
                return PipelineResult(
                    content="To rotate a deploy key...",
                    routed_to="knowledge",
                    trace_id="test-trace-001",
                )
            ...

        monkeypatch.setattr("app.srop.pipeline.run", mock_run)
    """
    call_state = {"count": 0}

    async def mock_run_adk(agent, user_id, session_id, user_message):
        call_state["count"] += 1
        message = user_message.lower()

        if any(term in message for term in ("poem", "joke", "story")):
            return (REFUSAL_TEXT, "smalltalk", [], [])

        if (
            "last agent" in message
            or "previous agent" in message
            or "last_agent" in message
            or ("last request" in message and "agent" in message)
        ):
            last_agent = _extract_last_agent(getattr(agent, "instruction", ""))
            return (
                f"Last agent was {last_agent}.",
                "smalltalk",
                [],
                [],
            )

        if "plan tier" in message or "account" in message:
            plan_tier = _extract_plan_tier(getattr(agent, "instruction", ""))
            return (
                f"Your plan tier is {plan_tier}.",
                "account",
                [
                    {
                        "tool_name": "get_account_status",
                        "args": {"user_id": user_id},
                        "result": {"user_id": user_id, "plan_tier": plan_tier},
                    }
                ],
                [],
            )

        if "rotate" in message or "deploy key" in message:
            return (
                "To rotate a deploy key, follow the docs steps. [chunk_test_001 score=0.92]",
                "knowledge",
                [
                    {
                        "tool_name": "search_docs",
                        "args": {"query": user_message, "k": 5},
                        "result": [
                            {
                                "chunk_id": "chunk_test_001",
                                "score": 0.92,
                                "content": "Rotate deploy key steps...",
                                "metadata": {"source": "deploy-keys.md"},
                            }
                        ],
                    }
                ],
                ["chunk_test_001"],
            )

        return ("Hello! How can I help?", "smalltalk", [], [])

    monkeypatch.setattr("app.srop.pipeline._run_adk", mock_run_adk)
    return call_state


def _extract_plan_tier(instruction: str) -> str:
    for line in instruction.splitlines():
        if line.strip().startswith("- plan_tier:"):
            return line.split(":", 1)[1].strip() or "free"
    return "free"


def _extract_last_agent(instruction: str) -> str:
    for line in instruction.splitlines():
        if line.strip().startswith("- last_agent:"):
            return line.split(":", 1)[1].strip() or "unknown"
    return "unknown"
