# =============================================================================
# app/tools/document_tool.py — ChromaDB Retrieval as a LangChain Tool
# =============================================================================
#
# 🧠 LEARNING NOTE — Why Expose RAG as a Tool?
#
# In the RAG Agent (rag_agent.py), the retrieval is built into the chain.
# But the Tool Agent uses a different pattern: ReAct (Reason + Act).
#
# ReAct pattern:
#   1. THINK  → "The user is asking about vacation policy. I should check the docs."
#   2. ACT    → Call the document_retrieval tool with "vacation policy"
#   3. OBSERVE → Read the retrieved document chunks
#   4. THINK  → "Based on the docs, the answer is 15 days per year."
#   5. RESPOND → Generate final answer
#
# By exposing ChromaDB retrieval as a tool, the Tool Agent can:
#   • Decide WHEN to retrieve (vs. use its parametric knowledge)
#   • Combine retrieval with other tools (e.g., retrieve docs + search web)
#   • Query different collections for different parts of a question
#
# This creates a more flexible agent than the dedicated RAG Agent,
# but at the cost of more LLM calls (ReAct loop overhead).
# =============================================================================

from typing import Optional

from langchain_core.tools import tool

from app.config import settings
from app.utils.logger import get_logger
from app.vectorstore.chroma_client import get_vector_store

logger = get_logger(__name__)


@tool
def document_retrieval_tool(query: str, collection_name: Optional[str] = None, k: int = 4) -> str:
    """
    Retrieves relevant document chunks from the knowledge base for a given query.

    Use this tool when the user's question might be answered by internal documents,
    company policies, product manuals, or any other stored knowledge base content.
    Do NOT use this for general knowledge or current events — use web_search instead.

    Args:
        query: The search query to find relevant documents. Be specific and use
               keywords that would appear in the relevant documents.
        collection_name: Optional ChromaDB collection to search. Leave empty to
                        use the default collection.
        k: Number of document chunks to retrieve (default: 4, max: 10).

    Returns:
        Formatted string of relevant document chunks with source citations,
        or a message indicating no relevant documents were found.
    """
    resolved_collection = collection_name or settings.chroma_collection_name
    k = min(k, 10)  # Cap at 10 to prevent excessive context

    logger.info(
        "document_retrieval_tool_called",
        query=query,
        collection=resolved_collection,
        k=k,
    )

    try:
        vector_store = get_vector_store(collection_name=resolved_collection)
        docs = vector_store.similarity_search(query, k=k)

        if not docs:
            return (
                f"No relevant documents found in the '{resolved_collection}' collection "
                f"for the query: '{query}'. "
                "The knowledge base may not contain information about this topic."
            )

        # Format results for LLM consumption
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            page_str = f" | Page {page + 1}" if page is not None else ""
            parts.append(
                f"[Document {i} — Source: {source}{page_str}]\n{doc.page_content}"
            )

        result = "\n\n---\n\n".join(parts)

        logger.info(
            "document_retrieval_complete",
            query=query,
            docs_found=len(docs),
        )

        return result

    except Exception as e:
        logger.error("document_retrieval_failed", query=query, error=str(e))
        return f"Error retrieving documents: {str(e)}"
