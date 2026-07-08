# =============================================================================
# app/llm/openai_llm.py — OpenAI LLM Provider
# =============================================================================
#
# 🧠 LEARNING NOTE — What is ChatOpenAI?
#
# ChatOpenAI is LangChain's wrapper around OpenAI's Chat Completions API.
# It gives us the same `BaseChatModel` interface as ChatBedrock — meaning
# the rest of the app doesn't need to know WHICH provider it's talking to.
#
# Supported models (as of 2025):
#   • gpt-4o             → most capable, best for agents and RAG
#   • gpt-4o-mini        → faster and cheaper, good for simple tasks
#   • gpt-3.5-turbo      → legacy, fastest and cheapest
#
# Authentication:
#   ChatOpenAI reads OPENAI_API_KEY from the environment automatically.
#   You just need it set in your .env file — no explicit passing needed.
#
# OpenAI vs Bedrock — when to use which:
#   • OpenAI  → Easier setup, cutting-edge GPT models, great for prototyping
#   • Bedrock → Stays within AWS, no data leaves your AWS account,
#               good for compliance (HIPAA, SOC2), cost via AWS billing
# =============================================================================

from functools import lru_cache
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def get_openai_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    streaming: bool = False,
) -> BaseChatModel:
    """
    Creates and returns a cached OpenAI LLM instance.

    🧠 LEARNING NOTE — Parameters:
      model       → GPT model name (defaults to settings.openai_llm_model)
      temperature → 0.0 = factual/consistent, 1.0 = creative/varied
      max_tokens  → max response length in tokens (~4 chars per token)
      streaming   → yields tokens as they arrive (great for chat UIs)

    Returns:
      BaseChatModel — same interface as Bedrock LLM, fully interchangeable.
    """
    resolved_model = model or settings.openai_llm_model

    logger.info(
        "initializing_openai_llm",
        model=resolved_model,
        temperature=temperature,
        streaming=streaming,
    )

    return ChatOpenAI(
        # ChatOpenAI automatically reads OPENAI_API_KEY from the environment.
        # You can also pass it explicitly: api_key=settings.openai_api_key
        api_key=settings.openai_api_key,
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        # timeout → fail fast instead of hanging indefinitely
        timeout=60,
        # max_retries → automatically retry on rate limit / transient errors
        max_retries=2,
    )
