# =============================================================================
# app/chains/rag_chain.py — Retrieval-Augmented Generation (RAG) Chain
# =============================================================================
#
# 🧠 LEARNING NOTE — What is a RAG Chain?
#
# RAG = Retrieval-Augmented Generation.
#
# Instead of relying on the LLM's training data alone, we:
#   1. RETRIEVE relevant documents from ChromaDB (based on the user's question)
#   2. AUGMENT the LLM prompt with those retrieved documents as context
#   3. GENERATE an answer that is grounded in the provided documents
#
# This solves the "hallucination" problem — the LLM can only use facts
# from the retrieved documents, so it can't make things up (as easily).
#
# The Chain Architecture:
#
#   User Question
#       │
#       ▼
#   [Retriever] ─── ChromaDB similarity_search ──► [Relevant Chunks]
#       │
#       ▼
#   [Prompt Template] ← Combines: system + context chunks + question
#       │
#       ▼
#   [LLM] ─── Generates answer grounded in context
#       │
#       ▼
#   [Output Parser] ─── Extracts string answer from LLM response
#       │
#       ▼
#   Answer + Source Documents
#
# LangChain Expression Language (LCEL):
#   We use LCEL (the | pipe operator) to compose the chain.
#   LCEL chains are:
#     • Lazy — they don't execute until you call .invoke()
#     • Composable — each | connects a Runnable to the next
#     • Async-ready — use .ainvoke() for async execution
# =============================================================================

from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from app.config import settings
from app.llm.router import get_llm
from app.utils.logger import get_logger
from app.vectorstore.chroma_client import get_vector_store

logger = get_logger(__name__)

# =============================================================================
# RAG System Prompt
# =============================================================================
#
# 🧠 LEARNING NOTE — Prompt Engineering for RAG:
#
# The system prompt is critical for RAG quality. Key instructions:
#   1. Tell the LLM to ONLY use the provided context
#   2. Tell it to say "I don't know" if context is insufficient
#   3. Ask it to be concise and factual
#   4. Ask it to cite sources when possible
#
# {context} → replaced with retrieved document chunks
# {question} → replaced with the user's question
# =============================================================================
RAG_SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base.
Answer the user's question using ONLY the information provided in the context below.

Rules:
- If the context does not contain enough information to answer, say "I don't have enough information in the knowledge base to answer this question."
- Do not make up information or use knowledge outside the provided context.
- Be concise and factual.
- When citing information, reference the source document when available.

Context:
{context}
"""


def _format_docs(docs: list[Document]) -> str:
    """
    Formats retrieved document chunks into a single string for the prompt.

    🧠 LEARNING NOTE:
    We format the context as:
      [Source: filename.pdf | Page: 1]
      Document content here...
      ----
      [Source: other_doc.txt]
      More content here...

    This helps the LLM understand which information came from where,
    enabling it to cite sources in its answer.

    Parameters:
      docs → list of Document objects from ChromaDB retrieval

    Returns:
      Formatted string to inject into {context} in the prompt
    """
    if not docs:
        return "No relevant documents found."

    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", None)

        header = f"[Source: {source}"
        if page is not None:
            header += f" | Page: {page + 1}"  # Convert 0-indexed to 1-indexed
        header += "]"

        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n----\n\n".join(parts)


def get_rag_chain(
    provider: str = "bedrock",
    collection_name: Optional[str] = None,
    k: int = 4,
):
    """
    Builds and returns a LangChain RAG chain.

    🧠 LEARNING NOTE — LCEL Chain Composition:

    The chain is built with the pipe operator (|):

      retrieval_chain | prompt | llm | output_parser

    Where:
      retrieval_chain  → { "context": retriever | format_docs, "question": passthrough }
      prompt           → ChatPromptTemplate (system + human messages)
      llm              → ChatBedrock or ChatOpenAI
      output_parser    → StrOutputParser (extracts string from LLM response)

    RunnableParallel({ "context": ..., "question": ... }) runs both branches
    in parallel and merges results into a dict that the prompt can use.

    RunnablePassthrough() just passes the input through unchanged.

    Parameters:
      provider        → "bedrock" or "openai"
      collection_name → ChromaDB collection to retrieve from
      k               → number of document chunks to retrieve (default: 4)

    Returns:
      A LangChain Runnable chain ready to be invoked with .invoke({"question": "..."})
    """
    llm = get_llm(provider)

    vector_store = get_vector_store(collection_name=collection_name)

    # -------------------------------------------------------------------------
    # Retriever: wraps VectorStore with similarity search
    # search_type="similarity" → cosine similarity search
    # search_kwargs={"k": k}   → return top-k results
    #
    # 🧠 LEARNING NOTE — Other retriever types:
    #   • "mmr" (Maximal Marginal Relevance) → balances relevance + diversity
    #     Avoids returning k near-identical chunks from the same section.
    #   • "similarity_score_threshold" → only return docs above a score threshold
    # -------------------------------------------------------------------------
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # Prompt with system context + human question
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    # -------------------------------------------------------------------------
    # LCEL Chain:
    #
    # Step 1 — RunnableParallel:
    #   Takes the input (the question string) and runs two things in parallel:
    #   a) Retrieves relevant docs → formats them as a string → becomes "context"
    #   b) Passes the question through unchanged → becomes "question"
    #   Result: {"context": "formatted docs...", "question": "user question"}
    #
    # Step 2 — prompt:
    #   Fills {context} and {question} placeholders in the prompt template.
    #   Result: A ChatPromptValue (list of messages ready for the LLM)
    #
    # Step 3 — llm:
    #   Sends the prompt to the LLM and gets an AIMessage back.
    #
    # Step 4 — StrOutputParser:
    #   Extracts the plain string content from AIMessage.
    # -------------------------------------------------------------------------
    rag_chain = (
        RunnableParallel(
            {
                "context": retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info(
        "rag_chain_created",
        provider=provider,
        collection=collection_name or settings.chroma_collection_name,
        k=k,
    )

    return rag_chain, retriever


async def rag_query(
    question: str,
    provider: str = "bedrock",
    collection_name: Optional[str] = None,
    k: int = 4,
) -> dict[str, Any]:
    """
    Executes a RAG query: retrieves relevant docs and generates an answer.

    This is the main function called by the RAG agent or the /chat endpoint.

    Parameters:
      question        → the user's question
      provider        → LLM provider ("bedrock" or "openai")
      collection_name → ChromaDB collection to search
      k               → number of documents to retrieve

    Returns:
      dict with keys:
        answer   → str — the AI-generated answer
        sources  → list[dict] — source document info for citations
        provider → str — which LLM was used
    """
    logger.info(
        "rag_query_started",
        question_length=len(question),
        provider=provider,
        collection=collection_name or settings.chroma_collection_name,
    )

    try:
        chain, retriever = get_rag_chain(
            provider=provider,
            collection_name=collection_name,
            k=k,
        )

        # Run retriever and chain in parallel (retriever also runs inside chain,
        # but we run it separately here to get source documents for the response)
        retrieved_docs = await retriever.ainvoke(question)
        answer = await chain.ainvoke(question)

        # Build source info for the response
        sources = []
        for doc in retrieved_docs:
            sources.append(
                {
                    "content": doc.page_content[:500],  # truncate for response
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page"),
                    "metadata": {
                        k: v
                        for k, v in doc.metadata.items()
                        if k not in ("source", "page")
                    },
                }
            )

        logger.info(
            "rag_query_complete",
            answer_length=len(answer),
            sources_count=len(sources),
        )

        return {
            "answer": answer,
            "sources": sources,
            "provider": provider,
        }

    except Exception as e:
        logger.error("rag_query_failed", error=str(e), exc_info=True)
        raise
