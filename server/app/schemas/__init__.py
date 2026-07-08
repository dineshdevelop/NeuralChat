# =============================================================================
# app/schemas/__init__.py — Pydantic Schema Exports
# =============================================================================
# Centralizes all schema imports for easy access.
# Usage: from app.schemas import ChatRequest, ChatResponse, IngestResponse

from app.schemas.chat import ChatRequest, ChatResponse, SourceDocument
from app.schemas.ingest import IngestRequest, IngestResponse, CollectionStats
from app.schemas.auth import (
    TokenRequest,
    TokenResponse,
    APIKeyCreateRequest,
    APIKeyResponse,
    CurrentUserResponse,
)

__all__ = [
    # Chat
    "ChatRequest",
    "ChatResponse",
    "SourceDocument",
    # Ingest
    "IngestRequest",
    "IngestResponse",
    "CollectionStats",
    # Auth
    "TokenRequest",
    "TokenResponse",
    "APIKeyCreateRequest",
    "APIKeyResponse",
    "CurrentUserResponse",
]
