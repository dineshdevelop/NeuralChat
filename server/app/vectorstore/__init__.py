# app/vectorstore/__init__.py
# Makes 'vectorstore' a Python package.
# Exposes the main helpers at the package level for convenience.
from app.vectorstore.chroma_client import get_vector_store, get_chroma_client
from app.vectorstore.embeddings import get_embedding_model

__all__ = ["get_vector_store", "get_chroma_client", "get_embedding_model"]
