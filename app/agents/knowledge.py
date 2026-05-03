"""
KnowledgeAgent — answers docs questions using RAG.

Uses search_docs to retrieve relevant chunks and cites chunk IDs with scores.
"""
from google.adk.agents import LlmAgent

from app.agents.tools.search_docs import search_docs
from app.settings import settings

KNOWLEDGE_INSTRUCTION = """
You answer Helix product questions using the search_docs tool.
Always call search_docs for docs questions. Do not answer without retrieval.

Citations are required for every factual claim using the format:
[chunk_id score=0.xx]

If search_docs returns no relevant chunks, say you could not find evidence.
User context (user_id, plan_tier, last_agent) will be provided in the system message.
"""

knowledge_agent = LlmAgent(
    name="knowledge",
    model=settings.adk_model,
    instruction=KNOWLEDGE_INSTRUCTION,
    tools=[search_docs],
)
