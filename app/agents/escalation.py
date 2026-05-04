"""
EscalationAgent — opens support tickets via create_ticket.
"""
from google.adk.agents import LlmAgent

from app.agents.tools.ticket_tools import create_ticket
from app.settings import settings
from app.srop.state import SessionState

ESCALATION_INSTRUCTION = """
You handle support escalations and ticket creation.
Use the create_ticket tool when the user wants to open a support ticket,
request escalation, or asks for human follow-up.

Always call create_ticket when creating a ticket. Use the user_id from the
system message. If the user does not specify priority, default to normal.

If the user asks about their last ticket or open tickets, respond using
last_ticket_id and open_ticket_ids from the system message. If none exist,
say there are no open tickets yet.
"""


def build_escalation_agent(state: SessionState) -> LlmAgent:
    instruction = _build_escalation_instruction(state)
    return LlmAgent(
        name="escalation",
        model=settings.adk_model,
        instruction=instruction,
        tools=[create_ticket],
    )


def _build_escalation_instruction(state: SessionState) -> str:
    return (
        f"{ESCALATION_INSTRUCTION}\n\n"
        "Current user context:\n"
        f"- user_id: {state.user_id}\n"
        f"- plan_tier: {state.plan_tier}\n"
        f"- last_ticket_id: {state.last_ticket_id}\n"
        f"- open_ticket_ids: {state.open_ticket_ids}\n"
        f"- last_agent: {state.last_agent}\n"
        f"- turn_count: {state.turn_count}\n"
    )

escalation_agent = LlmAgent(
    name="escalation",
    model=settings.adk_model,
    instruction=ESCALATION_INSTRUCTION,
    tools=[create_ticket],
)
