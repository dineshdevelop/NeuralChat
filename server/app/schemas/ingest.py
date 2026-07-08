# =============================================================================
# app/schemas/ingest.py — Document Ingestion Request / Response Models
# =============================================================================
#
# 🧠 LEARNING NOTE — File Upload in FastAPI:
#
# FastAPI handles file uploads via `python-multipart`.
# The request is multipart/form-data (not JSON) when uploading files.
#
# In the route handler, files are received as:
#   files: list[UploadFile] = File(...)
#
# IngestRequest is used for the *optional* form fields alongside the file upload.
# The full schema is documented here for clarity and OpenAPI docs generation.
#
# IngestResponse is a standard JSON response returned after ingestion.
# =============================================================================

from typing import Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """
    Metadata fields for POST /ingest (non-file fields).

    In FastAPI, file upload routes use Form() for text fields and File() for
    file fields. This model documents the expected shape of the form data.

    Note: The actual files are received as `UploadFile` objects in the route,
    not as part of this Pydantic model.
    """

    collection_name: str = Field(
        default="documents",
        max_length=128,
        description=(
            "ChromaDB collection to store the ingested documents in. "
            "Defaults to the server's default collection. "
            "Use separate collections for different document sets."
        ),
        examples=["hr_documents", "product_manuals"],
    )

    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=8000,
        description=(
            "Number of characters per text chunk. "
            "Smaller chunks = more precise retrieval but more chunks. "
            "Larger chunks = more context per chunk but less precise."
        ),
    )

    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description=(
            "Number of characters to overlap between adjacent chunks. "
            "Overlap prevents context from being cut at chunk boundaries."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "collection_name": "hr_documents",
                "chunk_size": 1000,
                "chunk_overlap": 200,
            }
        }


class IngestResponse(BaseModel):
    """
    Response body for POST /ingest.

    Returns details about what was ingested so the caller can verify the result.
    """

    status: str = Field(
        ...,
        description="Ingestion status — 'success' or 'partial_failure'.",
    )

    collection_name: str = Field(
        ...,
        description="The ChromaDB collection that documents were added to.",
    )

    files_processed: int = Field(
        ...,
        description="Number of files successfully processed.",
    )

    chunks_added: int = Field(
        ...,
        description=(
            "Total number of text chunks embedded and stored in ChromaDB. "
            "One file typically produces many chunks."
        ),
    )

    errors: list[str] = Field(
        default_factory=list,
        description="List of error messages for files that failed to process.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "collection_name": "hr_documents",
                "files_processed": 3,
                "chunks_added": 142,
                "errors": [],
            }
        }


class CollectionStats(BaseModel):
    """
    Stats about a ChromaDB collection.

    Used by admin endpoints to inspect collection contents.
    """

    name: str = Field(..., description="Collection name.")
    count: int = Field(..., description="Number of chunks stored.", ge=0)
    metadata: Optional[dict] = Field(
        default=None,
        description="Collection settings (e.g., distance metric).",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if stats could not be retrieved.",
    )
