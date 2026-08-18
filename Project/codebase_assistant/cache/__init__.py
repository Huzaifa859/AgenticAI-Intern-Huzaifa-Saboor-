"""
cache
=====

Cross-run caching utilities shared by the specialized agents.

Contains OutputCache, a persistent, per-repository cache of LLM output
for Documentation, Testing, and Code Analysis, mirroring the design of
the RAG index manifest (`rag/indexer.py`): a small JSON file next to
the vector store, keyed by a content hash so a source change
invalidates the cached entry automatically.
"""

from .output_cache import OutputCache

__all__ = ["OutputCache"]
