# =============================================================================
# app/config.py — Centralized Application Configuration
# =============================================================================
#
# 🧠 LEARNING NOTE:
# We use Pydantic's BaseSettings to manage all configuration.
#
# Why Pydantic Settings?
#   ✅ Automatically reads values from environment variables AND .env files
#   ✅ Validates types (e.g., ensures PORT is an int, not a string)
#   ✅ Raises clear errors if required variables are missing
#   ✅ Provides IDE autocomplete for all config values
#   ✅ One source of truth — import `settings` anywhere in the app
#
# How it works:
#   1. Python reads your .env file (via python-dotenv / pydantic-settings)
#   2. Each field in Settings maps to an env variable name
#   3. `lru_cache` ensures the config is only loaded ONCE (performance)
#
# Usage anywhere in the app:
#   from app.config import settings
#   print(settings.aws_region)
# =============================================================================

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    Pydantic BaseSettings automatically reads from:
      1. Environment variables (os.environ)
      2. A .env file (specified by env_file below)

    Field(...) means the value is REQUIRED — app will fail to start if missing.
    Field(default=...) means it has a safe default value.
    """

    # -------------------------------------------------------------------------
    # Pydantic Settings Config
    # env_file         → which file to read (can be a list for multiple files)
    # env_file_encoding → file encoding
    # case_sensitive   → False means AWS_REGION and aws_region both work
    # extra            → "ignore" means unknown env vars don't cause errors
    # -------------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # App Settings
    # -------------------------------------------------------------------------
    app_title: str = Field(default="Multi-Agent RAG Chatbot")
    app_version: str = Field(default="1.0.0")

    # Literal type means only these exact string values are allowed.
    # Pydantic will raise a validation error for anything else.
    app_env: Literal["development", "staging", "production"] = Field(
        default="development"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # CORS — which frontend origins can call our API
    # We store as a comma-separated string, then parse into a list below.
    cors_origins: str = Field(default="http://localhost:3000")

    @property
    def cors_origins_list(self) -> list[str]:
        """Converts the comma-separated string into a Python list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    # -------------------------------------------------------------------------
    # AWS Settings
    # -------------------------------------------------------------------------
    aws_access_key_id: str = Field(default="")
    aws_secret_access_key: str = Field(default="")
    aws_region: str = Field(default="us-east-1")

    # -------------------------------------------------------------------------
    # AWS Bedrock Model Settings
    # These are the model IDs from the Bedrock console.
    # -------------------------------------------------------------------------
    bedrock_llm_model_id: str = Field(
        default="amazon.nova-micro-v1:0"
    )
    bedrock_embed_model_id: str = Field(default="amazon.titan-embed-text-v2:0")

    # -------------------------------------------------------------------------
    # OpenAI Settings
    # -------------------------------------------------------------------------
    openai_api_key: str = Field(default="")
    openai_llm_model: str = Field(default="gpt-4o")
    openai_embed_model: str = Field(default="text-embedding-3-small")

    # -------------------------------------------------------------------------
    # LLM Provider
    # This controls which LLM is used by default.
    # Can be overridden per-request via the API body.
    # -------------------------------------------------------------------------
    default_llm_provider: Literal["bedrock", "openai"] = Field(default="bedrock")

    # -------------------------------------------------------------------------
    # ChromaDB Settings
    # -------------------------------------------------------------------------
    chroma_persist_dir: str = Field(default="./chroma_data")
    chroma_collection_name: str = Field(default="documents")

    # Which embedding model to use when indexing documents.
    embedding_provider: Literal["bedrock", "openai", "local"] = Field(
        default="bedrock"
    )

    # -------------------------------------------------------------------------
    # Cognito Settings (Authentication)
    # -------------------------------------------------------------------------
    cognito_user_pool_id: str = Field(default="")
    cognito_app_client_id: str = Field(default="")
    cognito_region: str = Field(default="us-east-1")

    @property
    def cognito_jwks_url(self) -> str:
        """
        Constructs the Cognito JWKS (JSON Web Key Set) URL.
        This URL is used to fetch the public keys that verify JWT signatures.

        Format: https://cognito-idp.{region}.amazonaws.com/{pool_id}/.well-known/jwks.json
        """
        return (
            f"https://cognito-idp.{self.cognito_region}.amazonaws.com"
            f"/{self.cognito_user_pool_id}/.well-known/jwks.json"
        )

    @property
    def cognito_issuer(self) -> str:
        """
        The JWT 'iss' (issuer) claim that Cognito puts in every token.
        We validate this claim to ensure tokens were issued by OUR user pool.
        """
        return (
            f"https://cognito-idp.{self.cognito_region}.amazonaws.com"
            f"/{self.cognito_user_pool_id}"
        )

    # -------------------------------------------------------------------------
    # API Key Auth Settings
    # -------------------------------------------------------------------------
    api_keys_file: str = Field(default="./api_keys.json")

    # -------------------------------------------------------------------------
    # Rate Limiting
    # -------------------------------------------------------------------------
    rate_limit_per_minute: int = Field(default=60)

    # -------------------------------------------------------------------------
    # Tavily Web Search
    # -------------------------------------------------------------------------
    tavily_api_key: str = Field(default="")

    # -------------------------------------------------------------------------
    # Validators
    # @field_validator runs AFTER the field is parsed from env.
    # We use it to warn if critical keys are missing in production.
    # -------------------------------------------------------------------------
    @field_validator("openai_api_key", mode="before")
    @classmethod
    def warn_missing_openai_key(cls, v: str) -> str:
        """Warn (don't fail) if OpenAI key is missing — user might use Bedrock."""
        if not v:
            import warnings
            warnings.warn(
                "OPENAI_API_KEY is not set. OpenAI provider will not work.",
                stacklevel=2,
            )
        return v


# =============================================================================
# Singleton Pattern with lru_cache
# =============================================================================
#
# 🧠 LEARNING NOTE:
# lru_cache(maxsize=1) means this function is only called ONCE.
# Every subsequent call returns the cached instance.
#
# This is the standard FastAPI pattern for dependency injection of settings.
# It avoids re-reading the .env file on every API request.
#
# Usage:
#   from app.config import get_settings
#   settings = get_settings()
#
# Or use the pre-imported singleton below for convenience:
#   from app.config import settings
# =============================================================================
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a cached singleton instance of Settings."""
    return Settings()


# Pre-imported singleton — use this for simple imports across the codebase.
settings = get_settings()
