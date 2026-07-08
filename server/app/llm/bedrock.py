# =============================================================================
# app/llm/bedrock.py — AWS Bedrock LLM Provider
# =============================================================================
#
# 🧠 LEARNING NOTE — What is AWS Bedrock?
#
# AWS Bedrock is a managed service that gives you access to foundation models
# (like Claude, Titan, Llama) via a simple API — without managing any servers.
#
# Key concepts:
#   • Model ID   → unique identifier for each model (e.g., anthropic.claude-3-5-sonnet...)
#   • Region     → Bedrock availability varies by region (us-east-1 is most supported)
#   • IAM Role   → in production, ECS task role grants Bedrock access (no keys needed)
#   • Streaming  → Bedrock supports streaming responses (tokens appear as they generate)
#
# How LangChain connects to Bedrock:
#   LangChain's `ChatBedrock` wraps boto3's Bedrock client.
#   It handles: request formatting, response parsing, retry logic, streaming.
#
# boto3 credential chain (in order of priority):
#   1. Explicit keys (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in env)
#   2. IAM Role (ECS task role, EC2 instance profile)
#   3. AWS SSO / CLI profile
#   → You don't need to pass credentials explicitly if using IAM Roles!
# =============================================================================

from functools import lru_cache
from typing import Optional

from langchain_aws import ChatBedrock
from langchain_core.language_models import BaseChatModel

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def get_bedrock_llm(
    model_id: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    streaming: bool = False,
) -> BaseChatModel:
    """
    Creates and returns a cached AWS Bedrock LLM instance.

    🧠 LEARNING NOTE — Parameters:
      model_id    → which foundation model to use (defaults to settings value)
      temperature → controls randomness (0.0 = deterministic, 1.0 = creative)
                    For RAG/factual tasks: use 0.0
                    For creative writing: use 0.7-0.9
      max_tokens  → maximum length of the model's response
      streaming   → if True, tokens are yielded one by one (for real-time UI)

    🧠 LEARNING NOTE — lru_cache:
      @lru_cache means the same function call with the same args returns the
      SAME object — we don't create a new LLM instance on every request.
      This is important for performance (boto3 session setup has overhead).

    Returns:
      A LangChain BaseChatModel — a standard interface that works the same
      way regardless of the underlying model (Bedrock, OpenAI, etc.).
      This lets the rest of the code be provider-agnostic.
    """
    resolved_model_id = "amazon.nova-micro-v1:0"

    logger.info(
        "initializing_bedrock_llm",
        model_id=resolved_model_id,
        temperature=temperature,
        streaming=streaming,
    )

    # -------------------------------------------------------------------------
    # Model-specific kwargs
    # Different models have different parameter names:
    #   • Anthropic Claude → max_tokens (in model_kwargs)
    #   • Amazon Titan     → maxTokenCount (in model_kwargs)
    # LangChain's ChatBedrock handles these differences automatically.
    # -------------------------------------------------------------------------
    model_kwargs: dict = {}

    if "anthropic" in resolved_model_id:
        # Claude models accept anthropic_version for API compatibility
        model_kwargs = {
            "max_tokens": max_tokens,
            "anthropic_version": "bedrock-2023-05-31",
        }
    elif "titan" in resolved_model_id.lower():
        model_kwargs = {
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
            }
        }

    # -------------------------------------------------------------------------
    # ChatBedrock initialization
    # region_name       → must match where your Bedrock models are enabled
    # model_id          → the specific model to call
    # model_kwargs      → model-specific generation parameters
    # streaming         → enables token-by-token streaming
    # -------------------------------------------------------------------------
    return ChatBedrock(
        region_name=settings.aws_region,
        model_id=resolved_model_id,
        model_kwargs=model_kwargs,
        streaming=streaming,
    )
