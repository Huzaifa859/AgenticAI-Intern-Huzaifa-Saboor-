"""
vectordb.py
===========

Placeholder for the vector database backing the RAG pipeline: storing
and querying embedded chunks.

No real vector store integration yet (no ChromaDB or any other
vector DB backend).
"""

from __future__ import annotations


class VectorDB:
    """
    Stores and queries embedded chunks for retrieval.
    """

    def query(self, embedding, top_k: int = 5):
        """
        Query the vector store for the most similar embeddings.

        Args:
            embedding: Query embedding vector.
            top_k: Maximum number of results to return.

        TODO: Implement real vector store integration (e.g. ChromaDB,
        FAISS, or another backend) for storage and similarity search.
        """
        # TODO: implement real vector store query
        pass
