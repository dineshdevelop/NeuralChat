# =============================================================================
# app/vectorstore/chroma_client.py — ChromaDB Client & Collection Manager
# =============================================================================
#
# 🧠 LEARNING NOTE — What is ChromaDB?
#
# ChromaDB is an open-source, AI-native vector database.
# Think of it like a database, but instead of querying by exact match,
# you query by SIMILARITY — "find me documents similar to this question."
#
# Core concepts:
#   • Client     → the connection to ChromaDB (like a DB connection)
#   • Collection → like a table — groups related embeddings together
#                  e.g., one collection per document set or topic
#   • Document   → the text chunk stored with its embedding vector
#   • Embedding  → the vector representation of the document
#   • Metadata   → extra info stored alongside (filename, page, date, etc.)
#
# Two ChromaDB modes:
#   1. PersistentClient (local) → saves data to disk in CHROMA_PERSIST_DIR
#      ✅ Simple, no server needed, great for dev
#      ❌ Only one process can write at a time
#
#   2. HttpClient (server mode) → connects to a ChromaDB server process
#      ✅ Multiple processes can write simultaneously
#      ✅ Can be deployed as a Docker container
#      ❌ Requires running a separate server
#
# We use PersistentClient for now (local dev). In production, you'd switch
# to HttpClient pointing to a ChromaDB server in Docker/ECS.
#
# LangChain Integration:
#   We use LangChain's `Chroma` wrapper which combines:
#     • The ChromaDB client (for storage)
#     • An embedding model (for converting text to vectors)
#     • A retriever interface (for querying from LangChain chains)
# =============================================================================

from functools import lru_cache
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from app.config import settings
from app.vectorstore.embeddings import get_embedding_model
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    """
    Creates and returns a cached ChromaDB PersistentClient.

    🧠 LEARNING NOTE:
    The PersistentClient stores all data in the CHROMA_PERSIST_DIR directory.
    On startup, it loads existing data from disk automatically.
    On shutdown, data is already persisted (no manual save needed).

    We cache this with @lru_cache so the same client object is reused
    across all requests — avoids repeated disk I/O on every API call.

    Returns:
      chromadb.PersistentClient — low-level ChromaDB client.
      (Most of the time you'll use get_vector_store() instead.)
    """
    logger.info(
        "initializing_chroma_client",
        persist_dir=settings.chroma_persist_dir,
    )

    # chromadb.Settings controls behavior of the ChromaDB client.
    # anonymized_telemetry=False → disables usage data sent to ChromaDB servers.
    client_settings = chromadb.Settings(
        anonymized_telemetry=False,
        allow_reset=True,  # allow_reset=True lets us reset the DB in tests
    )

    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=client_settings,
    )


def get_vector_store(
    collection_name: Optional[str] = None,
    embedding_model: Optional[Embeddings] = None,
) -> VectorStore:
    """
    Returns a LangChain Chroma VectorStore for a given collection.

    🧠 LEARNING NOTE — VectorStore vs ChromaDB Client:
      ChromaDB Client → raw database operations (create collection, add docs)
      VectorStore     → LangChain abstraction with RAG-friendly methods:
        • similarity_search(query, k=4)  → find top-k similar docs
        • as_retriever()                 → plug into a LangChain chain
        • add_documents(docs)            → embed and store documents

    Parameters:
      collection_name  → which ChromaDB collection to use
                         Defaults to settings.chroma_collection_name
      embedding_model  → which embedding model to use for this collection
                         Defaults to the configured embedding model

    Returns:
      Chroma VectorStore — ready to use in RAG chains and agents.
    """
    resolved_collection = collection_name or settings.chroma_collection_name
    resolved_embeddings = embedding_model or get_embedding_model()

    logger.info(
        "getting_vector_store",
        collection=resolved_collection,
    )

    # -------------------------------------------------------------------------
    # LangChain's Chroma wrapper connects the ChromaDB client with
    # an embedding model, giving us a high-level VectorStore interface.
    # -------------------------------------------------------------------------
    return Chroma(
        client=get_chroma_client(),
        collection_name=resolved_collection,
        embedding_function=resolved_embeddings,
        # collection_metadata → optional settings for the collection
        # hnsw:space → distance metric: "cosine" (best for text), "l2", "ip"
        collection_metadata={"hnsw:space": "cosine"},
    )


def list_collections() -> list[str]:
    """
    Lists all existing ChromaDB collections.

    🧠 LEARNING NOTE:
    You can have multiple collections for different document sets.
    For example:
      • "hr_documents"      → company HR policies
      • "product_manuals"   → product documentation
      • "support_tickets"   → past support conversations

    Each collection has its own embeddings — they're completely independent.
    """
    client = get_chroma_client()
    collections = client.list_collections()
    names = [col.name for col in collections]
    logger.info("listed_collections", count=len(names), collections=names)
    return names


def delete_collection(collection_name: str) -> bool:
    """
    Deletes a ChromaDB collection and all its documents.

    ⚠️ WARNING: This is irreversible! All embeddings in the collection
    will be permanently deleted.

    Returns:
      True if deleted successfully, False if collection didn't exist.
    """
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
        logger.info("collection_deleted", collection=collection_name)
        return True
    except ValueError:
        logger.warning("collection_not_found", collection=collection_name)
        return False


def get_collection_stats(collection_name: Optional[str] = None) -> dict:
    """
    Returns stats about a ChromaDB collection.

    Returns a dict with:
      • name       → collection name
      • count      → number of documents stored
      • metadata   → collection settings (e.g., distance metric)
    """
    resolved_collection = collection_name or settings.chroma_collection_name
    client = get_chroma_client()

    try:
        collection = client.get_collection(resolved_collection)
        stats = {
            "name": resolved_collection,
            "count": collection.count(),
            "metadata": collection.metadata,
        }
        logger.info("collection_stats", **stats)
        return stats
    except Exception as e:
        logger.error(
            "collection_stats_failed",
            collection=resolved_collection,
            error=str(e),
        )
        return {"name": resolved_collection, "count": 0, "error": str(e)}
