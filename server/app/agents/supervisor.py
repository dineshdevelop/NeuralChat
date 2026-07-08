# =============================================================================
# app/agents/supervisor.py — LangGraph Supervisor / Router Agent
# =============================================================================
#
# 🧠 LEARNING NOTE — Multi-Agent Supervisor Pattern:
#
# The Supervisor is the "traffic controller" of the multi-agent system.
# It receives the user's message and decides which specialized agent should handle it:
#
#   ┌─────────────────────────────────────────────────────────┐
#   │                     SUPERVISOR                          │
#   │                                                         │
#   │  "What do our HR documents say about vacation policy?"  │
#   │       └──► rag_agent  (document question)               │
#   │                                                         │
#   │  "What is sqrt(2048) + today's top AI news?"            │
#   │       └──► tool_agent (needs calculator + web search)   │
#   │                                                         │
#   │  "Hi! Can you explain machine learning briefly?"        │
#   │       └──► conversation_agent (general knowledge)       │
#   └─────────────────────────────────────────────────────────┘
#
# Implementation:
#   We use a structured LLM output to get the routing decision.
#   The supervisor LLM outputs a JSON object: {"next": "rag_agent"}
#   LangGraph's conditional_edge reads this and routes accordingly.
#
# Graph Structure:
#
#   START
#     │
#     ▼
#   supervisor_node ──► rag_agent_node ──────────┐
#         │         ──► tool_agent_node ──────────┤──► END
#         │         ──► conversation_agent_node ──┘
#
# 🧠 LEARNING NOTE — Why LangGraph over plain chains?
#
# LangGraph gives us:
#   ✅ Stateful execution — state persists across all nodes
#   ✅ Conditional routing — dynamic paths based on LLM decisions
#   ✅ Cycles allowed — agents can loop (ReAct pattern)
#   ✅ Human-in-the-loop — can pause and wait for user input at any node
#   ✅ Built-in checkpointing — save and resume graph execution
#   ✅ Observability — each step is tracked and inspectable
# =============================================================================

from functools import lru_cache
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from app.agents.conversation_agent import conversation_agent_node
from app.agents.rag_agent import rag_agent_node
from app.agents.state import AgentState
from app.agents.tool_agent import tool_agent_node
from app.llm.router import get_llm
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Valid agent routes — Literal type for strict validation
AgentRoute = Literal["rag_agent", "tool_agent", "conversation_agent"]

# =============================================================================
# Supervisor Routing Prompt
# =============================================================================
#
# This prompt is the heart of the supervisor — it tells the LLM how to decide
# which agent should handle each query.
#
# Prompt engineering tips for routing:
#   • Be VERY explicit about each agent's purpose
#   • Give concrete examples of what each agent handles
#   • Keep the output format strict (JSON) to avoid parsing errors
# =============================================================================
SUPERVISOR_SYSTEM_PROMPT = """You are a routing supervisor for a multi-agent AI system.
Your ONLY job is to analyze the user's message and decide which specialized agent should handle it.

Available agents:
1. rag_agent — Use for questions about SPECIFIC DOCUMENTS or INTERNAL KNOWLEDGE:
   - Company policies, procedures, HR documents
   - Product documentation, manuals, specifications
   - Any question that starts with "According to...", "What does [document] say about..."
   - Questions about stored/uploaded documents in our knowledge base

2. tool_agent — Use for questions requiring COMPUTATION or CURRENT INFORMATION:
   - Any mathematical calculation ("What is 15% of 4500?", "Calculate sqrt(144)")
   - Current events, news, live data ("What's the latest news about...?")
   - Questions that need both retrieval AND calculation
   - Requests to search the web for up-to-date information

3. conversation_agent — Use for GENERAL CHAT and KNOWLEDGE questions:
   - Greetings and small talk ("Hi!", "Thank you", "How are you?")
   - General knowledge questions the LLM knows ("What is Python?", "Explain ML")
   - Follow-up questions and clarifications on previous responses
   - Anything that doesn't clearly need documents or tools

Rules:
- Respond with ONLY a valid JSON object: {"next": "<agent_name>"}
- Do NOT include any other text, explanation, or markdown
- If unsure between rag_agent and conversation_agent, prefer rag_agent
- If the query involves ANY calculation, always use tool_agent

Examples:
User: "What is the company vacation policy?" → {"next": "rag_agent"}
User: "Calculate 15% of 87500" → {"next": "tool_agent"}
User: "What's the latest news about OpenAI?" → {"next": "tool_agent"}
User: "Hi, how are you?" → {"next": "conversation_agent"}
User: "What is machine learning?" → {"next": "conversation_agent"}
User: "Search the knowledge base for product specs" → {"next": "rag_agent"}
"""


async def supervisor_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node function for the supervisor.

    Analyzes the user's message and decides which agent should handle it.
    Sets state["next_agent"] to the chosen agent name.

    🧠 LEARNING NOTE — Structured Output:
    We use with_structured_output() to get the LLM to return a specific
    Python dict format. This is more reliable than parsing free-form text.

    Alternatively, we parse the JSON manually (used here for broader
    LLM compatibility — Bedrock models vary in structured output support).

    Parameters:
      state → AgentState with messages, session_id, provider

    Returns:
      dict with updated next_agent field
    """
    import json

    message_obj = state["messages"][-1] if state["messages"] else ""
    message_text = getattr(message_obj, "content", str(message_obj))
    provider = state.get("provider", "bedrock")
    session_id = state.get("session_id", "default")

    logger.info(
        "supervisor_routing",
        session_id=session_id,
        message_preview=message_text[:100],
        provider=provider,
    )

    try:
        llm = get_llm(provider=provider, temperature=0.0)

        # Build the routing prompt
        routing_messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=f"Route this user message: {message}"),
        ]

        # Call the LLM for routing decision
        response = await llm.ainvoke(routing_messages)
        response_text = response.content.strip()

        # Parse the JSON routing decision
        # Handle cases where the LLM wraps JSON in markdown code blocks
        if "```" in response_text:
            # Extract JSON from markdown code block
            lines = response_text.split("\n")
            json_lines = [
                l for l in lines
                if not l.strip().startswith("```") and l.strip()
            ]
            response_text = " ".join(json_lines)

        routing_decision = json.loads(response_text)
        next_agent = routing_decision.get("next", "conversation_agent")

        # Validate the chosen agent is one we support
        valid_agents = {"rag_agent", "tool_agent", "conversation_agent"}
        if next_agent not in valid_agents:
            logger.warning(
                "supervisor_invalid_route",
                chosen=next_agent,
                fallback="conversation_agent",
            )
            next_agent = "conversation_agent"

        logger.info(
            "supervisor_decision",
            session_id=session_id,
            next_agent=next_agent,
        )

        return {"next_agent": next_agent}

    except (json.JSONDecodeError, KeyError) as e:
        # JSON parsing failed — fall back to conversation agent
        logger.warning(
            "supervisor_parse_failed",
            error=str(e),
            fallback="conversation_agent",
        )
        return {"next_agent": "conversation_agent"}

    except Exception as e:
        logger.error("supervisor_failed", error=str(e), exc_info=True)
        return {"next_agent": "conversation_agent"}


def _route_to_agent(state: AgentState) -> str:
    """
    Conditional edge function that reads next_agent and returns the node name.

    🧠 LEARNING NOTE — Conditional Edges in LangGraph:
    add_conditional_edges(source, condition_fn, path_map) adds a routing edge.
    The condition_fn receives state and returns a STRING key.
    The path_map maps that key to the next node name.

    Since we return the node name directly (no path_map needed), we just
    return the string that matches the registered node name.

    Parameters:
      state → current AgentState

    Returns:
      Node name string: "rag_agent_node", "tool_agent_node", or "conversation_agent_node"
    """
    next_agent = state.get("next_agent", "conversation_agent")

    routing_map = {
        "rag_agent":          "rag_agent_node",
        "tool_agent":         "tool_agent_node",
        "conversation_agent": "conversation_agent_node",
    }

    return routing_map.get(next_agent, "conversation_agent_node")


def build_agent_graph() -> StateGraph:
    """
    Builds and compiles the multi-agent LangGraph StateGraph.

    🧠 LEARNING NOTE — StateGraph Construction:

    1. Create a StateGraph with our AgentState schema.
    2. Add nodes (one per agent function).
    3. Add edges (connections between nodes).
       - Normal edge: always goes from A to B
       - Conditional edge: goes to A, B, or C based on state
    4. Set entry point (START → first node).
    5. Compile: validates the graph and returns a runnable.

    Graph visualization:

      START
        │
        ▼
      supervisor_node
        │
        ├──(rag_agent)──────────► rag_agent_node ──────────► END
        ├──(tool_agent)─────────► tool_agent_node ────────► END
        └──(conversation_agent)─► conversation_agent_node ─► END

    Returns:
      Compiled LangGraph (a Runnable ready to be invoked)
    """
    # Create the graph with our shared state schema
    graph = StateGraph(AgentState)

    # -------------------------------------------------------------------------
    # Register Nodes
    # Each node is a function: (state: AgentState) -> dict[str, Any]
    # -------------------------------------------------------------------------
    graph.add_node("supervisor_node",          supervisor_node)
    graph.add_node("rag_agent_node",           rag_agent_node)
    graph.add_node("tool_agent_node",          tool_agent_node)
    graph.add_node("conversation_agent_node",  conversation_agent_node)

    # -------------------------------------------------------------------------
    # Entry Point: START → supervisor
    # -------------------------------------------------------------------------
    graph.add_edge(START, "supervisor_node")

    # -------------------------------------------------------------------------
    # Conditional Routing: supervisor → one of the three sub-agents
    #
    # add_conditional_edges(source, path_fn, path_map=None)
    #   • source   → the node AFTER which we route
    #   • path_fn  → function(state) → str key
    #   • If path_map is None, the returned string IS the next node name
    # -------------------------------------------------------------------------
    graph.add_conditional_edges(
        "supervisor_node",
        _route_to_agent,
        {
            "rag_agent_node":          "rag_agent_node",
            "tool_agent_node":         "tool_agent_node",
            "conversation_agent_node": "conversation_agent_node",
        },
    )

    # -------------------------------------------------------------------------
    # Terminal Edges: each sub-agent → END
    # -------------------------------------------------------------------------
    graph.add_edge("rag_agent_node",          END)
    graph.add_edge("tool_agent_node",         END)
    graph.add_edge("conversation_agent_node", END)

    # Compile validates the graph structure and returns a runnable
    compiled = graph.compile()

    logger.info("agent_graph_compiled", nodes=["supervisor", "rag", "tool", "conversation"])

    return compiled


@lru_cache(maxsize=1)
def get_agent_graph():
    """
    Returns a cached compiled agent graph.

    🧠 LEARNING NOTE:
    Building the graph creates LLM clients and validates the structure.
    We cache it so it's only built once per server startup.
    """
    return build_agent_graph()


async def run_agent(
    message: str,
    session_id: str = "default",
    provider: str = "bedrock",
    collection_name: str | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """
    Main entry point to run the multi-agent system.

    This is what the /chat endpoint calls. It:
      1. Initializes the AgentState with the user's message
      2. Runs the compiled graph (supervisor → sub-agent → result)
      3. Returns a clean response dict

    Parameters:
      message         → the user's message/question
      session_id      → unique identifier for the conversation
      provider        → "bedrock" or "openai"
      collection_name → optional ChromaDB collection override
      metadata        → optional extra context (user info, auth claims)

    Returns:
      dict with keys:
        response    → str — the final answer
        agent_used  → str — which agent produced the answer
        sources     → list[dict] — source docs (RAG only)
        provider    → str — LLM provider used
        session_id  → str — echo of session ID
        error       → str | None — error message if something failed
    """
    from langchain_core.messages import HumanMessage as LCHumanMessage

    logger.info(
        "run_agent_called",
        session_id=session_id,
        provider=provider,
        message_length=len(message),
    )

    graph = get_agent_graph()

    # Initialize the AgentState
    initial_state: AgentState = {
        "messages":       [LCHumanMessage(content=message)],
        "session_id":     session_id,
        "provider":       provider,
        "collection_name": collection_name,
        "next_agent":     "",
        "final_answer":   "",
        "sources":        [],
        "agent_used":     "unknown",
        "error":          None,
        "metadata":       metadata or {},
    }

    try:
        # ainvoke runs the entire graph asynchronously
        final_state = await graph.ainvoke(initial_state)

        logger.info(
            "run_agent_complete",
            session_id=session_id,
            agent_used=final_state.get("agent_used"),
            answer_length=len(final_state.get("final_answer", "")),
        )

        return {
            "response":   final_state.get("final_answer", "No response generated."),
            "agent_used": final_state.get("agent_used", "unknown"),
            "sources":    final_state.get("sources", []),
            "provider":   provider,
            "session_id": session_id,
            "error":      final_state.get("error"),
        }

    except Exception as e:
        logger.error("run_agent_failed", session_id=session_id, error=str(e), exc_info=True)
        return {
            "response":   "I'm sorry, an unexpected error occurred. Please try again.",
            "agent_used": "unknown",
            "sources":    [],
            "provider":   provider,
            "session_id": session_id,
            "error":      str(e),
        }
