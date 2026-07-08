# =============================================================================
# app/schemas/chat.py — Chat Request / Response Models
# =============================================================================
#
# 🧠 LEARNING NOTE — Pydantic v2 Models:
#
# Pydantic models serve two purposes here:
#   1. REQUEST validation  → FastAPI validates incoming JSON against these models.
#      If the client sends the wrong type or missing field, FastAPI auto-returns
#      HTTP 422 Unprocessable Entity with a clear error message.
#
#   2. RESPONSE serialization → FastAPI uses these to serialize our Python
#      objects into clean JSON for the client.
#
# Key Pydantic v2 differences from v1:
#   • model_config replaces class Config
#   • Field(examples=[...]) replaces Field(example=...)
#   • model_validator / field_validator replace @validator
# =============================================================================

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request body for POST /chat.

    The client sends a JSON object matching this schema.
    FastAPI automatically validates and parses it into a ChatRequest instance.

    Example request body:
        {
            "message": "What is the company vacation policy?",
            "session_id": "user-123-session-abc",
            "provider": "bedrock",
            "collection_name": "hr_documents"
        }
    """

    message: str = Field(
        ...,  # required — no default value
        min_length=1,
        max_length=10_000,
        description="The user's chat message or question.",
        examples=["What is the company vacation policy?"],
    )

    session_id: str = Field(
        default="default",
        max_length=128,
        description=(
            "Unique session identifier for conversation history. "
            "Use the same session_id across multiple messages to maintain context."
        ),
        examples=["user-123-session-abc"],
    )

    provider: Literal["bedrock", "openai"] = Field(
        default="bedrock",
        description=(
            "LLM provider to use for this request. "
            "Defaults to the server's configured provider."
        ),
    )

    collection_name: Optional[str] = Field(
        default=None,
        max_length=128,
        description=(
            "ChromaDB collection to search for relevant documents. "
            "Defaults to the server's default collection."
        ),
        examples=["hr_documents"],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is the company vacation policy?",
                "session_id": "user-abc-session-001",
                "provider": "bedrock",
                "collection_name": "hr_documents",
            }
        }


class SourceDocument(BaseModel):
    """
    Represents a single source document chunk used as RAG context.

    Returned alongside the AI's answer so the user can verify the source.

    🧠 LEARNING NOTE:
    In RAG (Retrieval-Augmented Generation), the LLM answer is grounded
    in retrieved document chunks. Returning these sources enables:
      • Transparency — users can verify the AI's claims
      • Debugging    — developers can tune retrieval quality
      • Trust        — enterprise users expect citations
    """

    content: str = Field(
        ...,
        description="The text content of the retrieved document chunk.",
    )

    source: str = Field(
        default="unknown",
        description="File name or URL the document chunk came from.",
    )

    page: Optional[int] = Field(
        default=None,
        description="Page number in the source document (if applicable).",
    )

    score: Optional[float] = Field(
        default=None,
        description="Similarity score (0–1) — higher means more relevant.",
        ge=0.0,
        le=1.0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata from the document loader (chunk index, etc.).",
    )


class ChatResponse(BaseModel):
    """
    Response body for POST /chat.

    🧠 LEARNING NOTE — agent_used field:
    The multi-agent system can route to different sub-agents:
      • "rag_agent"          → used ChromaDB retrieval
      • "tool_agent"         → used tools (web search, calculator)
      • "conversation_agent" → general chat without retrieval

    Knowing which agent handled the request helps with debugging and
    explaining to the user why a certain answer was produced.
    """

    response: str = Field(
        ...,
        description="The AI-generated answer to the user's message.",
    )

    session_id: str = Field(
        ...,
        description="Echo of the session_id from the request.",
    )

    agent_used: str = Field(
        default="unknown",
        description="Which agent handled this request (rag_agent, tool_agent, conversation_agent).",
    )

    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Source document chunks used to generate the response (RAG only).",
    )

    provider: str = Field(
        ...,
        description="LLM provider that was used (bedrock or openai).",
    )

    user: Optional[str] = Field(
        default=None,
        description="The authenticated user's identifier (sub or email).",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "response": "The company vacation policy allows 15 days per year...",
                "session_id": "user-abc-session-001",
                "agent_used": "rag_agent",
                "sources": [
                    {
                        "content": "Employees are entitled to 15 days of paid vacation...",
                        "source": "hr_policy.pdf",
                        "page": 3,
                        "score": 0.92,
                        "metadata": {},
                    }
                ],
                "provider": "bedrock",
                "user": "user@example.com",
            }
        }
