# =============================================================================
# app/vectorstore/embeddings.py — Embedding Model Setup
# =============================================================================
#
# 🧠 LEARNING NOTE — What are Embeddings?
#
# Embeddings convert text into a list of numbers (a "vector").
# Similar texts produce similar vectors — this is what enables semantic search.
#
# Example:
#   "cat"   → [0.21, -0.45, 0.87, ...]   (768 or 1536 numbers)
#   "kitten"→ [0.19, -0.42, 0.84, ...]   (very similar → "close" in vector space)
#   "car"   → [-0.33, 0.91, -0.12, ...]  (very different)
#
# In our RAG pipeline:
#   1. At ingest time: we embed each document chunk and store in ChromaDB
#   2. At query time:  we embed the user's question
#   3. ChromaDB finds the stored vectors closest to the question vector
#   4. Those closest chunks = most relevant context for the LLM
#
# Three embedding providers:
#
#   🔵 Bedrock (Amazon Titan Embed):
#      - Runs in AWS, data stays in your account
#      - Good for compliance (HIPAA, SOC2, FedRAMP)
#      - Model: amazon.titan-embed-text-v2:0 (1536 dimensions)
#
#   🟢 OpenAI:
#      - Excellent quality, widely used
#      - Model: text-embedding-3-small (1536 dims, cheap + fast)
#      - Model: text-embedding-3-large (3072 dims, highest quality)
#
#   🟡 Local (sentence-transformers):
#      - Free, runs on your CPU/GPU — no API calls
#      - Slightly lower quality but great for development/testing
#      - Model: all-MiniLM-L6-v2 (384 dims, very fast)
# =============================================================================

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """
    Returns the appropriate embedding model based on settings.embedding_provider.

    🧠 LEARNING NOTE:
    The return type is `Embeddings` — a LangChain abstract base class.
    All three providers implement the same interface:
      model.embed_documents(["text1", "text2"])  → list of vectors
      model.embed_query("question text")         → single vector

    This means our ChromaDB and RAG code works identically regardless
    of which embedding provider is configured.

    Returns:
      Embeddings — the configured embedding model instance (cached).
    """
    provider = settings.embedding_provider
    logger.info("initializing_embedding_model", provider=provider)

    if provider == "bedrock":
        return _get_bedrock_embeddings()
    elif provider == "openai":
        return _get_openai_embeddings()
    elif provider == "local":
        return _get_local_embeddings()
    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. "
            f"Choose from: 'bedrock', 'openai', 'local'."
        )


def _get_bedrock_embeddings() -> Embeddings:
    """
    Creates AWS Bedrock Titan embedding model.

    🧠 LEARNING NOTE:
    BedrockEmbeddings uses boto3 under the hood.
    For local dev: needs AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY in .env
    For production: uses the ECS task's IAM Role automatically (no keys needed)
    """
    from langchain_aws import BedrockEmbeddings

    logger.info(
        "using_bedrock_embeddings",
        model_id=settings.bedrock_embed_model_id,
        region=settings.aws_region,
    )

    return BedrockEmbeddings(
        region_name=settings.aws_region,
        model_id=settings.bedrock_embed_model_id,
    )


def _get_openai_embeddings() -> Embeddings:
    """
    Creates OpenAI embedding model.

    🧠 LEARNING NOTE:
    text-embedding-3-small is the recommended default:
      - 1536 dimensions
      - Very fast and cheap ($0.02 per 1M tokens)
      - Great for most RAG use cases

    text-embedding-3-large (3072 dims) is better for high-stakes retrieval
    but costs more and is slower.
    """
    from langchain_openai import OpenAIEmbeddings

    logger.info(
        "using_openai_embeddings",
        model=settings.openai_embed_model,
    )

    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.openai_embed_model,
    )


def _get_local_embeddings() -> Embeddings:
    """
    Creates a local sentence-transformers embedding model.

    🧠 LEARNING NOTE:
    This runs entirely on your machine — no API calls, no cost.
    First run downloads the model (~90MB) from HuggingFace Hub.
    Subsequent runs use the cached model.

    all-MiniLM-L6-v2:
      - 384 dimensions (smaller than API models)
      - Very fast on CPU
      - Good quality for most tasks
      - Open source, MIT license
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    logger.info(
        "using_local_embeddings",
        model=model_name,
        note="Running locally — first run will download the model",
    )

    return HuggingFaceEmbeddings(
        model_name=model_name,
        # encode_kwargs control how texts are converted to vectors
        encode_kwargs={"normalize_embeddings": True},
    )
