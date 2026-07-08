# =============================================================================
# app/agents/tool_agent.py — Tool-Calling ReAct Agent Node
# =============================================================================
#
# 🧠 LEARNING NOTE — ReAct Pattern (Reason + Act):
#
# The ReAct pattern interleaves REASONING and ACTING:
#
#   Thought: "The user wants to know the square root of 2048 and current AI news"
#   Act:     Call calculator_tool("sqrt(2048)")
#   Observe: "45.254833995939045"
#   Thought: "Now I need the news"
#   Act:     Call web_search("latest AI news 2024")
#   Observe: "[Search results...]"
#   Thought: "I have both pieces of information. Let me compose the answer."
#   Response: "sqrt(2048) ≈ 45.25. In current AI news: ..."
#
# LangGraph Implementation:
#   We use create_react_agent() from langgraph.prebuilt.
#   This creates a pre-built graph with two nodes:
#     1. "agent" node → LLM decides which tool to call (if any)
#     2. "tools" node → executes the chosen tools and returns results
#
#   The graph loops between "agent" and "tools" until the LLM decides to stop
#   (no more tool calls → END).
#
# Available Tools for this Agent:
#   • calculator_tool      → math expressions
#   • web_search           → internet search (Tavily or DuckDuckGo)
#   • document_retrieval   → ChromaDB knowledge base search
#
# When is the Tool Agent used?
#   The Supervisor routes here when the query requires:
#   • Computation ("What is 15% of 4500?")
#   • Current events ("What's the latest news about...?")
#   • Combined retrieval + calculation
#   • Any question that benefits from tool augmentation
# =============================================================================

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from app.agents.state import AgentState
from app.llm.router import get_llm
from app.tools.calculator import calculator_tool
from app.tools.document_tool import document_retrieval_tool
from app.tools.web_search import get_web_search_tool
from app.utils.logger import get_logger

logger = get_logger(__name__)

# System prompt for the tool agent
TOOL_AGENT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to powerful tools.

Available tools:
- calculator_tool: For any mathematical calculations or numeric computations
- web_search: For finding current information, news, or facts from the internet
- document_retrieval_tool: For finding information in the internal knowledge base

Guidelines:
1. Use tools when they would improve answer accuracy or provide current information
2. For math problems, ALWAYS use the calculator tool rather than computing manually
3. For questions about recent events or live data, use web_search
4. For questions about company documents or internal policies, use document_retrieval_tool
5. You may chain multiple tool calls if needed
6. After gathering information, synthesize a clear, helpful response
7. Cite your sources when using retrieved information
"""


def _build_tool_agent(provider: str = "bedrock"):
    """
    Builds a ReAct tool-calling agent for the given LLM provider.

    🧠 LEARNING NOTE — create_react_agent():
    This is a pre-built LangGraph factory that creates a full agent graph:

      ┌─────────────────────────────────────────┐
      │  agent node → LLM with tools bound      │
      │       │                                  │
      │  ┌────▼────────────────────────────┐    │
      │  │ Tool calls? ─── YES → tools node│    │
      │  │              └─────────────────►│    │
      │  │ No tool calls? → END            │    │
      │  └─────────────────────────────────┘    │
      └─────────────────────────────────────────┘

    The agent loops until the LLM produces a response with no tool calls.

    Parameters:
      provider → "bedrock" or "openai"

    Returns:
      A compiled LangGraph agent (Runnable)
    """
    llm = get_llm(provider=provider, temperature=0.0)

    tools = [
        calculator_tool,
        document_retrieval_tool,
        get_web_search_tool(),
    ]

    # create_react_agent returns a compiled StateGraph
    agent = create_react_agent(
        model=llm,
        tools=tools,
        # prompt replaces the deprecated state_modifier in langgraph 0.2.x+
        prompt=SystemMessage(content=TOOL_AGENT_SYSTEM_PROMPT),
    )

    return agent, tools


async def tool_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node function for the tool-calling ReAct agent.

    Runs the ReAct loop: LLM reasons about tool calls → executes tools →
    observes results → decides to call more tools or produce a final answer.

    Parameters:
      state → current AgentState

    Returns:
      dict of state updates: final_answer, sources, agent_used, error
    """
    message = state["messages"][-1] if state["messages"] else ""
    provider = state.get("provider", "bedrock")
    session_id = state.get("session_id", "default")

    logger.info(
        "tool_agent_invoked",
        session_id=session_id,
        message_length=len(message),
        provider=provider,
    )

    try:
        agent, _tools = _build_tool_agent(provider=provider)

        # The ReAct agent expects messages in LangChain message format
        agent_input = {"messages": [HumanMessage(content=message)]}

        # ainvoke runs the full ReAct loop asynchronously
        result = await agent.ainvoke(agent_input)

        # Extract the final AI message from the result
        # result["messages"] contains all messages including tool calls/results
        final_message = None
        tool_calls_made = []

        for msg in reversed(result["messages"]):
            # Find the last AIMessage that has actual content (not a tool call)
            if hasattr(msg, "content") and msg.content:
                msg_type = type(msg).__name__
                if msg_type == "AIMessage":
                    final_message = msg.content
                    break

        # Collect tool call info for logging
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_made.append(tc.get("name", "unknown"))

        if not final_message:
            final_message = "I was unable to generate a response. Please try again."

        logger.info(
            "tool_agent_complete",
            session_id=session_id,
            tools_used=tool_calls_made,
            answer_length=len(final_message),
        )

        return {
            "final_answer": final_message,
            "sources": [],  # Tool results are embedded in the answer text
            "agent_used": f"tool_agent[{', '.join(tool_calls_made) or 'no tools'}]",
            "error": None,
        }

    except Exception as e:
        error_msg = f"Tool agent failed: {str(e)}"
        logger.error("tool_agent_failed", session_id=session_id, error=str(e))

        return {
            "final_answer": (
                "I encountered an error while using my tools. "
                "Please try rephrasing your question."
            ),
            "sources": [],
            "agent_used": "tool_agent",
            "error": error_msg,
        }
