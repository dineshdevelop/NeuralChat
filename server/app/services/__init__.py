# app/services/__init__.py
from app.services.ingestion_service import ingest_documents, ingest_from_directory

__all__ = ["ingest_documents", "ingest_from_directory"]
