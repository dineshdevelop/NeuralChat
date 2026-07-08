# =============================================================================
# app/agents/rag_agent.py — RAG Retrieval Agent Node
# =============================================================================
#
# 🧠 LEARNING NOTE — Agent vs Chain:
#
# A LangChain CHAIN is a fixed pipeline: input → step1 → step2 → output.
# A LangGraph AGENT is a node in a graph that:
#   • Receives the current state (AgentState dict)
#   • Performs its task (in this case: RAG retrieval + generation)
#   • Returns state updates to merge back into the graph state
#
# This RAG Agent node:
#   1. Takes the user's message from the graph state
#   2. Runs the RAG chain (retrieve → generate)
#   3. Updates the state with the answer + retrieved sources
#   4. Sets agent_used = "rag_agent" so the supervisor knows who answered
#
# When is the RAG Agent used?
#   The Supervisor routes here when the user's question is about
#   specific documents, company knowledge, or stored information.
#   Example: "What does the HR policy say about vacation days?"
# =============================================================================

from typing import Any

from app.agents.state import AgentState
from app.chains.rag_chain import rag_query
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def rag_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node function for the RAG agent.

    🧠 LEARNING NOTE — LangGraph Node Functions:
    A node function receives the current AgentState and returns a dict
    of state keys to UPDATE. LangGraph merges the returned dict into the
    existing state (partial update — you only return what changed).

    The state is a TypedDict (defined in state.py), so all updates
    are type-checked at development time.

    Parameters:
      state → current AgentState with at minimum: messages, session_id, provider

    Returns:
      dict of state updates: final_answer, sources, agent_used, error
    """
    message = state["messages"][-1] if state["messages"] else ""
    provider = state.get("provider", "bedrock")
    collection_name = state.get("collection_name")
    session_id = state.get("session_id", "default")

    logger.info(
        "rag_agent_invoked",
        session_id=session_id,
        message_length=len(message),
        provider=provider,
    )

    try:
        result = await rag_query(
            question=message,
            provider=provider,
            collection_name=collection_name,
            k=4,
        )

        logger.info(
            "rag_agent_complete",
            session_id=session_id,
            sources_count=len(result.get("sources", [])),
        )

        return {
            "final_answer": result["answer"],
            "sources": result["sources"],
            "agent_used": "rag_agent",
            "error": None,
        }

    except Exception as e:
        error_msg = f"RAG retrieval failed: {str(e)}"
        logger.error("rag_agent_failed", session_id=session_id, error=str(e))

        return {
            "final_answer": (
                "I encountered an error while searching the knowledge base. "
                "Please try again or rephrase your question."
            ),
            "sources": [],
            "agent_used": "rag_agent",
            "error": error_msg,
        }
