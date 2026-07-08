# =============================================================================
# app/llm/router.py — LLM Provider Router
# =============================================================================
#
# 🧠 LEARNING NOTE — Why a Router?
#
# Instead of hardcoding "use Bedrock" or "use OpenAI" in every agent and chain,
# we use a ROUTER (factory function) as the single place that decides.
#
# Benefits:
#   ✅ A/B testing — send 50% of traffic to Bedrock, 50% to OpenAI
#   ✅ Fallback    — if Bedrock fails, automatically try OpenAI
#   ✅ Per-request switching — API callers can pick their preferred model
#   ✅ Easy to add new providers (Anthropic direct, Cohere, etc.) later
#
# Design Pattern: Factory + Strategy
#   get_llm() is a FACTORY — it creates and returns the right object.
#   Each provider (Bedrock, OpenAI) is a STRATEGY — same interface, different impl.
#
# Usage:
#   from app.llm.router import get_llm
#
#   # Use the default provider from settings
#   llm = get_llm()
#
#   # Explicitly request a provider
#   llm = get_llm(provider="openai")
#
#   # With custom options
#   llm = get_llm(provider="bedrock", temperature=0.7, streaming=True)
# =============================================================================

from typing import Literal, Optional
from langchain_core.language_models import BaseChatModel

from app.config import settings
from app.llm.bedrock import get_bedrock_llm
from app.llm.openai_llm import get_openai_llm
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Type alias for valid provider names — used for IDE autocomplete + validation
LLMProvider = Literal["bedrock", "openai"]


def get_llm(
    provider: Optional[LLMProvider] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    streaming: bool = False,
    model_id: Optional[str] = None,
) -> BaseChatModel:
    """
    Factory function that returns the appropriate LLM based on the provider.

    🧠 LEARNING NOTE — Parameters:
      provider    → which LLM service to use ("bedrock" or "openai")
                    Defaults to settings.default_llm_provider if not specified.
      temperature → randomness level (0.0 = deterministic, 1.0 = very creative)
      max_tokens  → maximum tokens in the response
      streaming   → whether to stream tokens (True for real-time chat UIs)
      model_id    → optional override for the specific model (e.g., "gpt-4o-mini")

    Returns:
      BaseChatModel — a unified interface. Call it the same way regardless
      of whether it's Bedrock or OpenAI underneath:
        response = llm.invoke([HumanMessage(content="Hello!")])

    Raises:
      ValueError → if an unsupported provider is requested
    """

    # Use the configured default if no provider is explicitly requested
    resolved_provider: LLMProvider = provider or settings.default_llm_provider

    logger.info(
        "llm_provider_selected",
        provider=resolved_provider,
        temperature=temperature,
        streaming=streaming,
    )

    # -------------------------------------------------------------------------
    # Provider routing — dispatch to the right factory function.
    # We pass model_id only if explicitly provided (None = use defaults).
    # -------------------------------------------------------------------------
    if resolved_provider == "bedrock":
        return get_bedrock_llm(
            model_id=model_id,          # None → uses settings.bedrock_llm_model_id
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )

    elif resolved_provider == "openai":
        return get_openai_llm(
            model=model_id,             # None → uses settings.openai_llm_model
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )

    else:
        # This should never happen due to the Literal type, but good to be safe.
        raise ValueError(
            f"Unsupported LLM provider: '{resolved_provider}'. "
            f"Choose from: 'bedrock', 'openai'."
        )


def get_llm_with_fallback(
    primary: LLMProvider = "bedrock",
    fallback: LLMProvider = "openai",
    **kwargs,
) -> BaseChatModel:
    """
    Returns the primary LLM, falling back to the secondary if initialization fails.

    🧠 LEARNING NOTE — Fallback Pattern:
    In production, external APIs can fail. A fallback ensures your chatbot
    keeps working even if one provider has an outage.

    Usage:
        llm = get_llm_with_fallback(primary="bedrock", fallback="openai")
    """
    try:
        llm = get_llm(provider=primary, **kwargs)
        logger.info("llm_primary_selected", provider=primary)
        return llm
    except Exception as e:
        logger.warning(
            "llm_primary_failed_using_fallback",
            primary=primary,
            fallback=fallback,
            error=str(e),
        )
        return get_llm(provider=fallback, **kwargs)
