# =============================================================================
# app/tools/web_search.py — Web Search Tool via Tavily
# =============================================================================
#
# 🧠 LEARNING NOTE — Web Search in RAG Agents:
#
# Our RAG system answers questions from stored documents (ChromaDB).
# But what if the user asks about something NOT in our docs?
#   → "What's the latest news about AI?"
#   → "What is today's Python release?"
#
# A web search tool lets the agent query the live internet when the
# knowledge base doesn't have the answer.
#
# Tavily vs Google vs DuckDuckGo:
#   • Tavily   → purpose-built for AI agents. Returns clean, structured results
#                optimized for LLM consumption (not raw HTML). Has a free tier.
#   • Google   → requires a paid Custom Search API key + CSE ID.
#   • DuckDuckGo → free, no key needed, but rate-limited and less reliable.
#
# We implement both:
#   1. TavilySearchTool  → primary (if TAVILY_API_KEY is set)
#   2. DuckDuckGoSearchTool → fallback (no API key required)
#
# The agent gets whichever is available based on configuration.
# =============================================================================

from app.config import settings
from app.utils.logger import get_logger
from langchain_core.tools import BaseTool, tool

logger = get_logger(__name__)


def get_web_search_tool() -> BaseTool:
    """
    Returns the best available web search tool.

    Priority:
      1. Tavily (if TAVILY_API_KEY is configured)
      2. DuckDuckGo (no-key fallback)

    Returns:
      A LangChain Tool ready for use in agents.
    """
    if settings.tavily_api_key:
        logger.info("web_search_provider", provider="tavily")
        return _get_tavily_tool()
    else:
        logger.info("web_search_provider", provider="duckduckgo_fallback")
        return _get_duckduckgo_tool()


def _get_tavily_tool() -> BaseTool:
    """
    Creates a Tavily web search tool.

    🧠 LEARNING NOTE — TavilySearchResults:
    This is a LangChain-native tool that:
      • Calls the Tavily Search API
      • Returns max_results clean text snippets
      • Formats results optimally for LLM consumption
      • Supports include_raw_content=True for full page text

    Tavily pricing: 1000 free searches/month on the free tier.
    Sign up at: https://app.tavily.com
    """
    from langchain_community.tools.tavily_search import TavilySearchResults

    return TavilySearchResults(
        max_results=5,
        # include_answer=True → Tavily also provides a direct answer summary
        include_answer=True,
        # include_raw_content=False → only summaries (less tokens)
        include_raw_content=False,
        # search_depth="advanced" → more thorough but slower
        search_depth="basic",
        api_key=settings.tavily_api_key,
        name="web_search",
        description=(
            "Search the internet for current information, news, or facts not available "
            "in the knowledge base. Use this for recent events, live data, or general "
            "knowledge questions. Input should be a clear search query string."
        ),
    )


def _get_duckduckgo_tool() -> BaseTool:
    """
    Creates a DuckDuckGo search tool as a no-key fallback.

    🧠 LEARNING NOTE:
    DuckDuckGo's unofficial API is used here — no key required.
    It may be rate-limited or unavailable, so this is only a fallback.
    For production, always use Tavily or another paid provider.
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        return DuckDuckGoSearchRun(
            name="web_search",
            description=(
                "Search the internet for current information, news, or facts not available "
                "in the knowledge base. Use this for recent events, live data, or general "
                "knowledge questions. Input should be a clear search query string."
            ),
        )
    except ImportError:
        logger.warning(
            "duckduckgo_unavailable",
            note="Install duckduckgo-search: pip install duckduckgo-search",
        )
        # Return a stub tool that explains the situation
        return _get_search_unavailable_tool()


@tool
def _search_unavailable(query: str) -> str:
    """
    Placeholder tool when no search provider is available.

    Args:
        query: The search query (not used).

    Returns:
        An explanation that web search is not configured.
    """
    return (
        "Web search is not available. To enable it, set the TAVILY_API_KEY "
        "environment variable. Get a free key at https://app.tavily.com"
    )


def _get_search_unavailable_tool() -> BaseTool:
    """Returns a stub tool explaining that search is unavailable."""
    _search_unavailable.name = "web_search"
    _search_unavailable.description = (
        "Web search tool (currently unavailable — TAVILY_API_KEY not configured)"
    )
    return _search_unavailable
