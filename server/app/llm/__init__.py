# app/llm/__init__.py
# Makes 'llm' a Python package.
# Exposes the main router function at the package level for convenience.
from app.llm.router import get_llm, get_llm_with_fallback

__all__ = ["get_llm", "get_llm_with_fallback"]
