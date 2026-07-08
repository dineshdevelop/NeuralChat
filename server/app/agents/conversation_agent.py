# =============================================================================
# app/agents/conversation_agent.py — Conversational Agent Node
# =============================================================================
#
# 🧠 LEARNING NOTE — When is the Conversation Agent Used?
#
# The Supervisor routes here when the user's question is:
#   • General chat / small talk ("Hi!", "How are you?", "Thanks!")
#   • Follow-up questions on previous answers ("Tell me more", "Expand on that")
#   • Requests that don't need document retrieval or tools
#   • Questions where the LLM's parametric knowledge is sufficient
#     ("What is Python?", "Explain machine learning briefly")
#
# Unlike the RAG Agent, the Conversation Agent:
#   ✅ Maintains full conversation history (multi-turn aware)
#   ✅ Can use the LLM's built-in knowledge
#   ❌ Does NOT retrieve from ChromaDB
#   ❌ Does NOT call tools
#
# The in-memory session store from conversation_chain.py is reused here.
# Each unique session_id has its own conversation history.
# =============================================================================

from typing import Any

from app.agents.state import AgentState
from app.chains.conversation_chain import conversation_chat
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def conversation_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node function for the conversational agent.

    Handles general chat using the conversation chain with session memory.
    Returns the AI response and updates state with agent_used = "conversation_agent".

    Parameters:
      state → current AgentState

    Returns:
      dict of state updates: final_answer, sources (empty), agent_used, error
    """
    #message = state["messages"][-1] if state["messages"] else ""
    message_obj = state["messages"][-1] if state["messages"] else None
    message = message_obj.content if message_obj else ""
    provider = state.get("provider", "bedrock")
    session_id = state.get("session_id", "default")

    logger.info(
        "conversation_agent_invoked",
        session_id=session_id,
        message_length=len(message),
        provider=provider,
    )

    try:
        result = await conversation_chat(
            message=message,
            session_id=session_id,
            provider=provider,
        )

        logger.info(
            "conversation_agent_complete",
            session_id=session_id,
            history_length=result.get("history_length", 0),
        )

        return {
            "final_answer": result["answer"],
            "sources": [],  # No document sources for conversation
            "agent_used": "conversation_agent",
            "error": None,
        }

    except Exception as e:
        error_msg = f"Conversation agent failed: {str(e)}"
        logger.error("conversation_agent_failed", session_id=session_id, error=str(e))

        return {
            "final_answer": (
                "I'm sorry, I encountered an error generating a response. "
                "Please try again."
            ),
            "sources": [],
            "agent_used": "conversation_agent",
            "error": error_msg,
        }
