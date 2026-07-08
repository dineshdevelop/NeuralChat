# =============================================================================
# app/main.py — FastAPI Application Entry Point
# =============================================================================
#
# 🧠 LEARNING NOTE — Application Lifecycle:
#
# FastAPI uses "lifespan" context managers to run code at startup/shutdown.
# This is where we initialize expensive resources ONCE:
#   • ChromaDB client (disk I/O to load the database)
#   • LLM clients (boto3 session setup)
#   • JWKS cache for JWT verification
#
# Why not initialize in every request?
#   Creating a ChromaDB client or boto3 session takes ~100-500ms.
#   If we did it per-request, every API call would be slow.
#   By initializing once at startup, all requests share the same instances.
#
# The app is structured to grow across phases:
#   Phase 1: startup/shutdown + basic health endpoint
#   Phase 2: RAG ingestion endpoint
#   Phase 3: Agent chat endpoint
#   Phase 4: Auth endpoints
# =============================================================================

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils.logger import get_logger, setup_logging
from dotenv import load_dotenv

load_dotenv()
# Set up logging FIRST — before any other imports that might log
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    🧠 LEARNING NOTE:
    Everything BEFORE `yield` runs at startup.
    Everything AFTER `yield` runs at shutdown.

    This replaced the old @app.on_event("startup") decorator in FastAPI 0.95+.
    Using a context manager is cleaner because startup and shutdown logic
    live in the same function — easier to read and reason about.
    """
    # =========================================================================
    # STARTUP
    # =========================================================================
    logger.info(
        "application_starting",
        app=settings.app_title,
        version=settings.app_version,
        env=settings.app_env,
        llm_provider=settings.default_llm_provider,
    )

    # Initialize ChromaDB client at startup (warms up the disk connection)
    try:
        from app.vectorstore.chroma_client import get_chroma_client, list_collections
        client = get_chroma_client()
        collections = list_collections()
        logger.info(
            "chromadb_ready",
            persist_dir=settings.chroma_persist_dir,
            existing_collections=collections,
        )
    except Exception as e:
        logger.error("chromadb_startup_failed", error=str(e))
        # We don't raise here — app can still start without ChromaDB
        # (though RAG features won't work until it's fixed)

    # Initialize embedding model (downloads local model if needed)
    try:
        from app.vectorstore.embeddings import get_embedding_model
        get_embedding_model()
        logger.info(
            "embedding_model_ready",
            provider=settings.embedding_provider,
        )
    except Exception as e:
        logger.error("embedding_model_startup_failed", error=str(e))

    # Pre-compile the LangGraph agent graph (Phase 3)
    try:
        from app.agents.supervisor import get_agent_graph
        get_agent_graph()
        logger.info("agent_graph_ready")
    except Exception as e:
        logger.error("agent_graph_startup_failed", error=str(e))

    logger.info("application_ready", docs_url="/docs")

    # =========================================================================
    # Hand control to FastAPI — app is now running and serving requests
    # =========================================================================
    yield

    # =========================================================================
    # SHUTDOWN
    # =========================================================================
    logger.info("application_shutting_down")
    # ChromaDB PersistentClient auto-saves — no manual flush needed.
    # Close any other resources here in later phases (Redis, DB connections).
    logger.info("application_stopped")


# =============================================================================
# FastAPI App Instance
# =============================================================================
app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="""
    ## Multi-Agent RAG Chatbot

    A production-grade AI chatbot with:
    - 🤖 **Multi-agent orchestration** via LangGraph
    - 📚 **RAG (Retrieval-Augmented Generation)** via ChromaDB
    - 🔧 **Tool calling** (web search, calculator)
    - ☁️ **AWS Bedrock** (Claude 3.5, Titan) + **OpenAI** (GPT-4o)
    - 🔐 **Authentication** via API Key + AWS Cognito JWT
    """,
    # OpenAPI docs available at /docs (Swagger UI) and /redoc
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# =============================================================================
# Middleware
# =============================================================================

# CORS (Cross-Origin Resource Sharing)
# 🧠 LEARNING NOTE:
# Browsers block requests from different origins by default (security feature).
# CORS middleware tells the browser which origins are allowed to call our API.
# In dev: we allow localhost frontends.
# In prod: we'll restrict to our actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],       # Allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],       # Allow any headers (including X-API-Key, Authorization)
)


# =============================================================================
# Routes — Phase 1: Basic Health Endpoints
# (More routers will be mounted in later phases)
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint — returns basic API info.
    Useful to confirm the server is running.
    """
    return {
        "message": f"Welcome to {settings.app_title}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint — used by load balancers and monitoring tools.

    🧠 LEARNING NOTE:
    In AWS ECS, the Application Load Balancer (ALB) calls /health periodically.
    If this returns a non-200 status, ECS marks the task as unhealthy and
    replaces it with a new one automatically.

    In production, you'd also check:
      • Can we reach ChromaDB? (check disk)
      • Can we reach Bedrock? (check AWS connectivity)
      • Are all required env vars set?
    """
    from app.vectorstore.chroma_client import get_collection_stats

    # Check ChromaDB connectivity
    try:
        stats = get_collection_stats()
        chroma_status = "healthy"
        chroma_docs = stats.get("count", 0)
    except Exception as e:
        chroma_status = f"unhealthy: {str(e)}"
        chroma_docs = 0

    return {
        "status": "healthy",
        "environment": settings.app_env,
        "version": settings.app_version,
        "llm_provider": settings.default_llm_provider,
        "embedding_provider": settings.embedding_provider,
        "services": {
            "chromadb": {
                "status": chroma_status,
                "collection": settings.chroma_collection_name,
                "documents_indexed": chroma_docs,
            }
        },
    }

# =============================================================================
# Demo / Core Endpoints 
# =============================================================================

from app.schemas.chat import ChatRequest, ChatResponse
from app.agents.supervisor import run_agent
from fastapi import Depends
from app.api.deps import require_user_or_admin, require_admin
from app.auth.models import CurrentUser
from app.api.routes.auth import router as auth_router

app.include_router(auth_router)

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest, current_user: CurrentUser = require_user_or_admin):
    """
    Main chat endpoint.
    Wires up the LangGraph multi-agent system from Phase 3.
    """
    result = await run_agent(
        message=request.message,
        session_id=request.session_id,
        provider=request.provider,
        collection_name=request.collection_name
    )
    return ChatResponse(**result)

from fastapi import UploadFile, File
from typing import List, Optional
from app.services.ingestion_service import ingest_documents
from app.schemas.ingest import IngestResponse

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_endpoint(
    files: List[UploadFile] = File(...),
    collection_name: Optional[str] = None,
    current_user: CurrentUser = require_admin
):
    """
    Ingest endpoint to parse and store documents into ChromaDB.
    """
    return await ingest_documents(files, collection_name)