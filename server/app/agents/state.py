# =============================================================================
# app/agents/state.py — Shared LangGraph Agent State
# =============================================================================
#
# 🧠 LEARNING NOTE — LangGraph State:
#
# LangGraph is a stateful graph framework built on top of LangChain.
# Every node in the graph reads from and writes to a shared STATE object.
#
# The state is a TypedDict — a Python dict with type annotations.
# TypedDict gives us:
#   • Type safety (IDEs can catch errors)
#   • Clear documentation of what's in the state
#   • Works with LangGraph's state management system
#
# Reducers (Annotated with add_messages):
#   By default, state updates REPLACE existing values.
#   Some fields use reducers to control how updates are merged.
#   add_messages is a special reducer: instead of replacing the messages list,
#   it APPENDS new messages to the existing list.
#
# State Flow in our multi-agent system:
#
#   User Input
#       │
#       ▼
#   AgentState initialized
#   { messages: ["user question"],
#     session_id: "abc",
#     provider: "bedrock",
#     ... }
#       │
#       ▼
#   Supervisor Node (reads messages, writes next_agent)
#       │
#       ▼
#   Sub-Agent Node (reads messages, writes final_answer + sources)
#       │
#       ▼
#   END (final state returned to caller)
# =============================================================================

from typing import Any, Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state for the multi-agent LangGraph system.

    Every agent node reads from this state and returns a partial update dict.
    LangGraph merges the partial update into the full state after each node.

    Fields:
      messages       → conversation history (human + AI messages)
                       Uses add_messages reducer: new messages are APPENDED
      session_id     → unique conversation identifier
      provider       → which LLM to use ("bedrock" or "openai")
      collection_name → ChromaDB collection to search
      next_agent     → which sub-agent the supervisor selected
      final_answer   → the text answer to return to the user
      sources        → list of source document dicts (for RAG answers)
      agent_used     → which agent actually produced the final answer
      error          → error message if something went wrong (None if success)
      metadata       → optional extra context (user info, auth, etc.)
    """

    # conversation history — add_messages reducer appends instead of replacing
    messages: Annotated[list[Any], add_messages]

    # Session & request context
    session_id: str
    provider: str
    collection_name: Optional[str]

    # Routing — set by supervisor, read by conditional edges
    next_agent: str

    # Output — set by the winning sub-agent
    final_answer: str
    sources: list[dict]
    agent_used: str
    error: Optional[str]

    # Optional metadata (auth info, etc.)
    metadata: dict[str, Any]
