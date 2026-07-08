# =============================================================================
# app/utils/logger.py — Structured Logging Setup
# =============================================================================
#
# 🧠 LEARNING NOTE:
# We use TWO libraries together:
#
#   • structlog → adds structure to log messages (key=value pairs)
#                 Makes logs machine-readable (JSON) in production.
#
#   • rich      → makes log output beautiful in the terminal during development
#                 (colors, icons, syntax highlighting)
#
# Why structured logging?
#   Standard logging:  "User 123 sent a message"
#   Structured logging: {"event": "chat_request", "user_id": "123", "latency_ms": 45}
#
#   The structured version is:
#   ✅ Searchable in CloudWatch / Datadog / Grafana Loki
#   ✅ Filterable (e.g., show me all logs where latency_ms > 100)
#   ✅ Consistent format across all log lines
#
# How to use:
#   from app.utils.logger import get_logger
#   logger = get_logger(__name__)
#   logger.info("chat_request", user_id="abc", message_length=42)
# =============================================================================

import logging
import sys

import structlog
from rich.console import Console
from rich.logging import RichHandler

from app.config import settings


def setup_logging() -> None:
    """
    Configure structlog + Python's standard logging.

    Call this ONCE at application startup (in main.py).

    In development: pretty, colored output via Rich.
    In production:  JSON lines output for log aggregation tools.
    """

    # -------------------------------------------------------------------------
    # Step 1: Set the log level from settings.
    # LOG_LEVEL controls the minimum severity of messages to output.
    #   DEBUG   → everything (verbose, dev only)
    #   INFO    → normal operation messages
    #   WARNING → something unexpected but non-fatal
    #   ERROR   → something failed
    # -------------------------------------------------------------------------
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # -------------------------------------------------------------------------
    # Step 2: Configure the handler based on environment.
    # In development, use Rich's beautiful handler.
    # In production, use a plain StreamHandler (outputs JSON lines).
    # -------------------------------------------------------------------------
    if settings.app_env == "development":
        # Rich handler gives us colored output with tracebacks
        handler = RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            markup=True,
            show_path=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        # Plain handler — structlog will format as JSON
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))

    # Apply the handler to the root logger
    logging.basicConfig(
        level=log_level,
        handlers=[handler],
        format="%(message)s",
    )

    # Silence overly verbose third-party loggers
    for noisy_logger in ["httpx", "httpcore", "boto3", "botocore", "urllib3"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # -------------------------------------------------------------------------
    # Step 3: Configure structlog processors.
    # Processors are a pipeline that transforms log records before output.
    # Each processor receives (logger, method, event_dict) and returns event_dict.
    # -------------------------------------------------------------------------
    shared_processors: list = [
        # Adds the log level (INFO, DEBUG, etc.) to every log record
        structlog.stdlib.add_log_level,
        # Adds a timestamp in ISO 8601 format
        structlog.processors.TimeStamper(fmt="iso"),
        # Adds the logger name (usually __name__ of the calling module)
        structlog.stdlib.add_logger_name,
        # If an exception is being logged, formats its traceback
        structlog.processors.StackInfoRenderer(),
        # Renders exception info into the log dict
        structlog.dev.set_exc_info,
    ]

    if settings.app_env == "development":
        # In dev: human-readable colored output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        # In prod: JSON output — one JSON object per line
        # This is what CloudWatch, Datadog, etc. expect.
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        # Use the standard library's logging system as the backend
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        # cache_logger_on_first_use → performance optimization
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Returns a named structlog logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("event_name", key1="value1", key2=42)

    The __name__ argument automatically uses the current module's name,
    making it easy to find which file generated a log message.
    """
    return structlog.get_logger(name)
