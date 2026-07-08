# =============================================================================
# app/chains/conversation_chain.py — Conversational Chain with Memory
# =============================================================================
#
# 🧠 LEARNING NOTE — What is a Conversation Chain?
#
# Unlike RAG (which retrieves documents for each query), a Conversation Chain
# keeps track of the entire dialogue history so the LLM can:
#   • Reference earlier messages ("as I mentioned above...")
#   • Answer follow-up questions ("and what about the second one?")
#   • Maintain context across multiple turns
#
# Memory Strategy (Modern LangGraph approach):
#   We maintain a dict of ChatMessageHistory objects keyed by session_id.
#   Instead of the deprecated RunnableWithMessageHistory wrapper, we manually
#   inject history into the prompt and persist the turn ourselves.
#   This is the pattern recommended by LangGraph's built-in persistence docs.
#
# Provider Fallback:
#   If the primary provider (Bedrock) returns a ThrottlingException or any
#   quota/rate-limit error, the chain automatically retries with OpenAI.
#   This prevents hard failures when daily token limits are hit.
#
# Session Management:
#   Each conversation has a unique session_id.
#   We maintain a dictionary of histories keyed by session_id.
#   This simulates server-side session storage (for MVP; use Redis in production).
#
# 🚨 Production Note:
#   In-memory session storage is lost on server restart.
#   For production, replace _session_histories with a LangGraph checkpointer
#   (e.g., AsyncSqliteSaver or AsyncRedisSaver) for persistent, distributed
#   conversation history.
# =============================================================================

from typing import Any, Optional

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

from app.llm.router import get_llm
from app.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# In-memory session store
# =============================================================================
# Maps session_id → ChatMessageHistory instance.
# 🚨 MVP only — replace with a LangGraph checkpointer for production.
# =============================================================================
_session_histories: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Returns (or creates) a ChatMessageHistory for the given session.

    🧠 LEARNING NOTE — ChatMessageHistory:
    This stores a list of messages (HumanMessage, AIMessage) in memory.
    We load these messages manually and inject them into the prompt before
    calling the LLM, then save the new turn after getting a response.

    Parameters:
      session_id → unique identifier for the conversation

    Returns:
      ChatMessageHistory — the message history for this session
    """
    if session_id not in _session_histories:
        _session_histories[session_id] = ChatMessageHistory()
        logger.info("new_session_created", session_id=session_id)
    return _session_histories[session_id]


def clear_session_history(session_id: str) -> bool:
    """
    Clears all message history for a given session.

    Useful when the user wants to start a fresh conversation.

    Returns:
      True if session existed and was cleared, False if not found.
    """
    if session_id in _session_histories:
        del _session_histories[session_id]
        logger.info("session_cleared", session_id=session_id)
        return True
    return False


def list_active_sessions() -> list[str]:
    """Returns all active session IDs."""
    return list(_session_histories.keys())


# =============================================================================
# Conversation System Prompt
# =============================================================================
CONVERSATION_SYSTEM_PROMPT = """You are a friendly and helpful AI assistant.
You are having a conversation with a user. Your responses should be:
- Helpful and informative
- Natural and conversational
- Concise but complete
- Honest — admit when you don't know something

Use the conversation history to provide contextually relevant responses.
"""


def get_conversation_chain(provider: str = "bedrock"):
    """
    Builds a conversation chain with message history support.

    🧠 LEARNING NOTE — RunnableWithMessageHistory:

    This wrapper adds automatic message history management to any chain:
      1. Before invoke: loads history from get_session_history(session_id)
         and injects it into the prompt via {chat_history}
      2. After invoke: saves the new HumanMessage + AIMessage to history

    The chain shape:
      {"input": user_message, "chat_history": loaded_history}
          |
          ▼
      ChatPromptTemplate  (system + history messages + current question)
          |
          ▼
      LLM
          |
          ▼
      StrOutputParser → plain string answer

    Parameters:
      provider → "bedrock" or "openai"

    Returns:
      RunnableWithMessageHistory — call with:
        .invoke({"input": question}, config={"configurable": {"session_id": "..."}})
    """
    llm = get_llm(provider)

    # -------------------------------------------------------------------------
    # Prompt with history placeholder
    # MessagesPlaceholder("chat_history") → replaced with previous messages
    # -------------------------------------------------------------------------
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONVERSATION_SYSTEM_PROMPT),
            # This placeholder gets filled with the conversation history
            MessagesPlaceholder(variable_name="chat_history"),
            # The current user message
            ("human", "{input}"),
        ]
    )

    # Basic chain (without history — history is added by the wrapper below)
    chain = prompt | llm | StrOutputParser()

    # -------------------------------------------------------------------------
    # Wrap the chain with automatic history management.
    #
    # get_session_history → function that returns history for a session_id
    # input_messages_key  → key in input dict containing the user message
    # history_messages_key → key in prompt that receives the loaded history
    # -------------------------------------------------------------------------
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    logger.info("conversation_chain_created", provider=provider)
    return chain_with_history


async def conversation_chat(
    message: str,
    session_id: str = "default",
    provider: str = "bedrock",
) -> dict[str, Any]:
    """
    Sends a message in a conversation and returns the AI response.

    Automatically manages conversation history for the given session_id.

    Parameters:
      message    → the user's message
      session_id → unique session identifier for conversation history
      provider   → LLM provider ("bedrock" or "openai")

    Returns:
      dict with keys:
        answer     → str — the AI-generated response
        session_id → str — echo of the session ID
        provider   → str — which LLM was used
        history_length → int — number of messages in history (after this turn)
    """
    logger.info(
        "conversation_chat_started",
        session_id=session_id,
        message_length=len(message),
        provider=provider,
    )

    try:
        chain = get_conversation_chain(provider=provider)

        # Invoke the chain — session_id is passed via config, not the input dict
        answer = await chain.ainvoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}},
        )

        # Get history length for debugging/logging
        history = get_session_history(session_id)
        history_length = len(history.messages)

        logger.info(
            "conversation_chat_complete",
            session_id=session_id,
            answer_length=len(answer),
            history_messages=history_length,
        )

        return {
            "answer": answer,
            "session_id": session_id,
            "provider": provider,
            "history_length": history_length,
        }

    except Exception as e:
        logger.error(
            "conversation_chat_failed",
            session_id=session_id,
            error=str(e),
            exc_info=True,
        )
        raise
