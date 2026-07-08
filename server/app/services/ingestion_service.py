# =============================================================================
# app/services/ingestion_service.py — Document Loading, Splitting & Embedding
# =============================================================================
#
# 🧠 LEARNING NOTE — What is the Ingestion Pipeline?
#
# The RAG ingestion pipeline takes raw documents (PDF, DOCX, TXT) and:
#   1. LOADS them      → extract raw text from the file
#   2. SPLITS them     → break long text into smaller chunks
#   3. EMBEDS them     → convert each chunk to a vector (numbers)
#   4. STORES them     → save vectors + text to ChromaDB
#
# Why split into chunks?
#   LLMs have a limited context window (e.g., 100K tokens).
#   ChromaDB retrieves the TOP-K most relevant chunks, not the whole doc.
#   Smaller chunks = more targeted retrieval = better RAG performance.
#
# Why overlap chunks?
#   If a sentence is cut in the middle at a chunk boundary, the model
#   loses context. Overlapping adjacent chunks prevents this.
#
# RecursiveCharacterTextSplitter:
#   LangChain's recommended splitter. It tries to split on:
#     1. Paragraph breaks (\n\n)  — preferred split point
#     2. Newlines (\n)
#     3. Spaces
#     4. Characters (last resort)
#   This preserves sentence/paragraph structure better than naive character split.
#
# Supported document types:
#   • PDF   → PyPDFLoader (uses pypdf)
#   • DOCX  → Docx2txtLoader (uses docx2txt)
#   • TXT   → TextLoader
#   • MD    → TextLoader (Markdown treated as plain text)
# =============================================================================

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from app.config import settings
from app.utils.logger import get_logger
from app.vectorstore.chroma_client import get_vector_store

logger = get_logger(__name__)

# Supported file types and their corresponding LangChain loaders
SUPPORTED_EXTENSIONS: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def load_document(file_path: str) -> list[Document]:
    """
    Loads a document from disk and returns a list of LangChain Document objects.

    🧠 LEARNING NOTE — LangChain Document:
    A Document is a simple container with two fields:
      • page_content: str  → the raw text
      • metadata: dict     → source, page number, etc.

    Different loaders produce different metadata:
      • PyPDFLoader  → {"source": "file.pdf", "page": 0}
      • Docx2txtLoader → {"source": "file.docx"}
      • TextLoader   → {"source": "file.txt"}

    Parameters:
      file_path → absolute path to the document on disk

    Returns:
      list[Document] — one Document per page (PDF) or one per file (DOCX/TXT)

    Raises:
      ValueError  → unsupported file extension
      RuntimeError → loading failed
    """
    ext = Path(file_path).suffix.lower()
    loader_class = SUPPORTED_EXTENSIONS.get(ext)

    if not loader_class:
        supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: {supported}"
        )

    logger.info(
        "loading_document",
        file=file_path,
        loader=loader_class.__name__,
    )

    try:
        loader = loader_class(file_path)
        docs = loader.load()
        logger.info("document_loaded", file=file_path, pages=len(docs))
        return docs
    except Exception as e:
        logger.error("document_load_failed", file=file_path, error=str(e))
        raise RuntimeError(f"Failed to load '{Path(file_path).name}': {e}") from e


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Splits documents into smaller, overlapping chunks.

    🧠 LEARNING NOTE — Text Splitting Strategy:
    RecursiveCharacterTextSplitter is the recommended LangChain splitter.

    It splits on these separators IN ORDER:
      ["\n\n", "\n", " ", ""]

    This means it tries paragraph breaks first, then newlines, then words,
    then individual characters (only if nothing else works).

    chunk_size    = max characters per chunk
    chunk_overlap = shared characters between adjacent chunks

    Example with chunk_size=100, chunk_overlap=20:
      Chunk 1: chars 0–100
      Chunk 2: chars 80–180   (20 chars overlap with chunk 1)
      Chunk 3: chars 160–260  (20 chars overlap with chunk 2)

    Parameters:
      documents    → list of Document objects from load_document()
      chunk_size   → max characters per chunk (default: 1000)
      chunk_overlap → overlap between chunks (default: 200)

    Returns:
      list[Document] — many smaller chunks, each with inherited metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # length_function → how to measure chunk size (len = character count)
        length_function=len,
        # add_start_index → adds "start_index" to metadata so you know where
        # in the original document each chunk came from
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    logger.info(
        "documents_split",
        input_docs=len(documents),
        output_chunks=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return chunks


async def ingest_upload_file(
    upload_file: UploadFile,
    collection_name: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[int, Optional[str]]:
    """
    Processes a single FastAPI UploadFile: saves to temp file, loads, splits,
    embeds, and stores in ChromaDB.

    🧠 LEARNING NOTE — Why a temp file?
    LangChain document loaders work with file PATHS on disk.
    FastAPI UploadFile gives us an in-memory file-like object (SpooledTemporaryFile).
    We must save it to disk first so the loader can open it by path.

    Parameters:
      upload_file     → FastAPI UploadFile from the multipart form
      collection_name → ChromaDB collection to store in
      chunk_size      → characters per chunk
      chunk_overlap   → overlap between chunks

    Returns:
      (chunks_added, error_message)
        chunks_added   → number of chunks stored (0 if error)
        error_message  → None if success, error string if failed
    """
    original_name = upload_file.filename or "unknown"
    ext = Path(original_name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return 0, f"Unsupported file type '{ext}' for '{original_name}'"

    # -------------------------------------------------------------------------
    # Write the uploaded file to a temp location on disk.
    # delete=False → we manage cleanup manually after loading.
    # suffix=ext   → preserves the extension so loaders can detect the type.
    # -------------------------------------------------------------------------
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=ext, mode="wb"
        ) as tmp:
            content = await upload_file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(
            "upload_file_saved",
            original_name=original_name,
            tmp_path=tmp_path,
            size_bytes=len(content),
        )

        # Load the document from temp file
        docs = load_document(tmp_path)

        # Inject the original filename into metadata so it's stored in ChromaDB
        for doc in docs:
            doc.metadata["source"] = original_name

        # Split into chunks
        chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if not chunks:
            return 0, f"No text could be extracted from '{original_name}'"

        # Embed and store in ChromaDB
        vector_store = get_vector_store(collection_name=collection_name)
        vector_store.add_documents(chunks)

        logger.info(
            "file_ingested",
            file=original_name,
            chunks=len(chunks),
            collection=collection_name or settings.chroma_collection_name,
        )

        return len(chunks), None

    except ValueError as e:
        # Unsupported type / validation error
        return 0, str(e)

    except RuntimeError as e:
        # Loading failed
        return 0, str(e)

    except Exception as e:
        logger.error(
            "ingest_failed",
            file=original_name,
            error=str(e),
            exc_info=True,
        )
        return 0, f"Unexpected error processing '{original_name}': {e}"

    finally:
        # Always clean up the temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.debug("temp_file_cleaned", tmp_path=tmp_path)


async def ingest_documents(
    upload_files: list[UploadFile],
    collection_name: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> dict:
    """
    Orchestrates ingestion of multiple uploaded files.

    Processes each file independently — a single file failure doesn't
    stop the other files from being processed.

    Parameters:
      upload_files    → list of FastAPI UploadFile objects
      collection_name → ChromaDB collection name
      chunk_size      → characters per chunk
      chunk_overlap   → overlap between chunks

    Returns:
      dict with keys:
        status         → "success" or "partial_failure"
        files_processed → int
        chunks_added   → int
        errors         → list[str]
    """
    resolved_collection = collection_name or settings.chroma_collection_name
    total_chunks = 0
    errors: list[str] = []
    files_processed = 0

    logger.info(
        "ingestion_started",
        file_count=len(upload_files),
        collection=resolved_collection,
    )

    for upload_file in upload_files:
        chunks_added, error = await ingest_upload_file(
            upload_file=upload_file,
            collection_name=resolved_collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if error:
            errors.append(error)
        else:
            files_processed += 1
            total_chunks += chunks_added

    status = "success" if not errors else "partial_failure"

    logger.info(
        "ingestion_complete",
        status=status,
        files_processed=files_processed,
        total_chunks=total_chunks,
        errors=len(errors),
    )

    return {
        "status": status,
        "collection_name": resolved_collection,
        "files_processed": files_processed,
        "chunks_added": total_chunks,
        "errors": errors,
    }


def ingest_from_directory(
    directory_path: str,
    collection_name: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> dict:
    """
    Ingests all supported documents from a local directory.

    Useful for bulk ingestion during setup / seeding the vector store.

    Parameters:
      directory_path  → path to directory containing documents
      collection_name → ChromaDB collection name
      chunk_size      → characters per chunk
      chunk_overlap   → overlap between chunks

    Returns:
      dict with ingestion results (same shape as ingest_documents)
    """
    resolved_collection = collection_name or settings.chroma_collection_name
    directory = Path(directory_path)

    if not directory.exists():
        raise ValueError(f"Directory not found: {directory_path}")

    # Find all supported files (non-recursive; use rglob for subdirectories)
    files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.warning("no_supported_files", directory=directory_path)
        return {
            "status": "success",
            "collection_name": resolved_collection,
            "files_processed": 0,
            "chunks_added": 0,
            "errors": [],
        }

    logger.info(
        "directory_ingestion_started",
        directory=directory_path,
        file_count=len(files),
        collection=resolved_collection,
    )

    total_chunks = 0
    errors: list[str] = []
    files_processed = 0

    for file_path in files:
        try:
            docs = load_document(str(file_path))
            chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            if chunks:
                vector_store = get_vector_store(collection_name=resolved_collection)
                vector_store.add_documents(chunks)
                total_chunks += len(chunks)
                files_processed += 1
                logger.info(
                    "file_ingested",
                    file=file_path.name,
                    chunks=len(chunks),
                )
        except Exception as e:
            error_msg = f"Failed to ingest '{file_path.name}': {e}"
            errors.append(error_msg)
            logger.error("file_ingest_failed", file=file_path.name, error=str(e))

    return {
        "status": "success" if not errors else "partial_failure",
        "collection_name": resolved_collection,
        "files_processed": files_processed,
        "chunks_added": total_chunks,
        "errors": errors,
    }
