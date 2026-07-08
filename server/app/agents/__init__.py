# =============================================================================
# app/agents/__init__.py — Agents Package Exports
# =============================================================================
#
# Public API for the agents module.
# The main entry point for the multi-agent system is run_agent().
#
# Usage in routes:
#   from app.agents import run_agent
#   result = await run_agent(message="...", session_id="...", provider="bedrock")
# =============================================================================

from app.agents.state import AgentState
from app.agents.supervisor import run_agent, get_agent_graph, build_agent_graph
from app.agents.rag_agent import rag_agent_node
from app.agents.tool_agent import tool_agent_node
from app.agents.conversation_agent import conversation_agent_node

__all__ = [
    # Main entry point
    "run_agent",
    # Graph management
    "get_agent_graph",
    "build_agent_graph",
    # State
    "AgentState",
    # Individual node functions (for testing / custom graphs)
    "rag_agent_node",
    "tool_agent_node",
    "conversation_agent_node",
]
