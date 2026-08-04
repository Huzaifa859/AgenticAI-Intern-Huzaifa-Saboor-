"""
rag
===

Retrieval-Augmented Generation subsystem.

Contains:
- Indexer / Retriever: the higher-level, currently-stubbed pipeline
  entry points used by agents.
- Ingestor, Chunker, EmbeddingGenerator, VectorDB: the individual
  pipeline stages (ingest -> chunk -> embed -> store/query), each a
  placeholder with a single stub method.

NOTE: Embeddings and actual vector search are not implemented yet.
No ChromaDB, no sentence-transformers, no real embedding calls.
"""

from .chunker import Chunker
from .embeddings import EmbeddingGenerator
from .indexer import Indexer
from .ingest import Ingestor
from .retriever import Retriever
from .vectordb import VectorDB

__all__ = [
    "Indexer",
    "Retriever",
    "Ingestor",
    "Chunker",
    "EmbeddingGenerator",
    "VectorDB",
]
