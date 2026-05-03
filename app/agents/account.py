"""
AccountAgent — answers account and usage questions using internal tools.
"""
from google.adk.agents import LlmAgent

from app.agents.tools.account_tools import get_account_status, get_recent_builds
from app.settings import settings

ACCOUNT_INSTRUCTION = """
You answer account and usage questions by calling account tools.
Always call a tool; do not invent account data.
Use the user_id from the system message. If missing, ask for it.
"""

account_agent = LlmAgent(
    name="account",
    model=settings.adk_model,
    instruction=ACCOUNT_INSTRUCTION,
    tools=[get_recent_builds, get_account_status],
)
