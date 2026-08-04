"""
retriever.py
============

Semantic retrieval over the index the Indexer built.

The only work this class does itself is embedding the incoming query and
translating results into the shape agents expect. Finding nearest
neighbours belongs to VectorDB, producing vectors belongs to
EmbeddingGenerator, and populating the index belongs to the Indexer.

Two views of the same search are exposed on purpose. `search` returns
CodeChunk objects with their scores, which is what the analysis layer
needs -- GroundingChecker cannot verify a claim without the file path
and line range. `retrieve` flattens those into the RetrievedChunk schema
the agents were scaffolded against. Neither is a wrapper around a
different query; both call one core path.

The query must be embedded with the same model that produced the stored
vectors, or the distances are meaningless. That is why the embedding
model name lives in Config and why sharing an Indexer here is the safest
way to construct one.

TODO: `rerank` currently just orders by similarity. Real cross-encoder
re-ranking is a Week 7 item.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from ..config import Config
from ..schemas.schemas import CodeChunk, RetrievedChunk
from .embeddings import EmbeddingGenerator
from .indexer import Indexer
from .vectordb import SearchResult, VectorDB

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant context chunks from the vector store.

    Collaborators are injectable so the Retriever can share the vector
    store and the already-loaded embedding model with whatever built the
    index, rather than opening a second connection or loading a second
    copy of the weights.
    """

    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        config: Optional[Config] = None,
        vector_db: Optional[VectorDB] = None,
        embedder: Optional[EmbeddingGenerator] = None,
        indexer: Optional[Indexer] = None,
    ) -> None:
        """
        Initialize the Retriever.

        Nothing is loaded here. The Supervisor constructs a Retriever
        during startup, so opening the store or loading the embedding
        model is deferred until a query actually arrives.

        Args:
            vector_store_path: Directory the vector store persists to.
                Kept as the leading parameter because the Supervisor
                already passes it by name. Falls back to
                `Config.chroma_persist_directory`.
            config: Optional Config instance. A default is loaded when
                not supplied.
            vector_db: Optional VectorDB. Built on first use when
                omitted.
            embedder: Optional EmbeddingGenerator. Built on first use
                when omitted. Pass the instance that built the index to
                be certain both sides use the same model.
            indexer: Optional Indexer. When given, its vector store and
                its Config are both adopted, which is the simplest way
                to guarantee the Retriever reads the collection the
                Indexer wrote with the settings it wrote it under.
        """
        # Adopting the store without the config would be a trap: the
        # Retriever would read the right collection while taking
        # retrieval_top_k and the embedding model name from defaults
        # that may not match what built the index.
        if config is None and indexer is not None:
            config = indexer.config

        self.config = config or Config.load()

        if vector_store_path is None and indexer is not None:
            vector_store_path = indexer.vector_store_path
        self.vector_store_path = (
            vector_store_path or self.config.chroma_persist_directory
        )

        if vector_db is None and indexer is not None:
            vector_db = indexer.vector_db

        self._vector_db = vector_db
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Lazily built collaborators
    # ------------------------------------------------------------------

    @property
    def vector_db(self) -> VectorDB:
        """
        The vector store, opened on first use.

        Returns:
            The VectorDB pointed at the configured collection.
        """
        if self._vector_db is None:
            self._vector_db = VectorDB(
                config=self.config, persist_directory=self.vector_store_path
            )
        return self._vector_db

    @property
    def embedder(self) -> EmbeddingGenerator:
        """
        The embedding model, loaded on first use.

        Returns:
            An EmbeddingGenerator using `Config.embedding_model_name`.
            The model cache is process-wide, so this reuses the weights
            the Indexer loaded rather than loading a second copy.
        """
        if self._embedder is None:
            self._embedder = EmbeddingGenerator(config=self.config)
        return self._embedder

    # ------------------------------------------------------------------
    # Index state
    # ------------------------------------------------------------------

    def is_indexed(self) -> bool:
        """
        Report whether the index holds anything.

        Worth checking before treating an empty result as "nothing
        matched": an unindexed repository produces the same empty list,
        and the two call for completely different responses. The
        proposal's abstention path depends on telling them apart.

        Args:
            None.

        Returns:
            True if at least one chunk is stored.

        Raises:
            ProviderUnavailableError: If the store cannot be opened.
        """
        return self.vector_db.count() > 0

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Find the chunks most similar to a query, with full detail.

        The core retrieval path. `retrieve` is a view over this, not a
        separate query.

        Args:
            query: Natural language or code query.
            top_k: Maximum chunks to return. Defaults to
                `Config.retrieval_top_k`.
            where: Optional metadata filter, applied inside the search
                so it narrows the candidate set rather than trimming
                results afterwards. Post-filtering would silently return
                fewer than `top_k` hits.
            min_score: Optional similarity floor. Results below it are
                dropped, which is how a caller avoids feeding weak
                matches to a model as though they were evidence.

        Returns:
            Hits ordered by similarity, highest first, each carrying the
            CodeChunk, its distance, and its score.

        Raises:
            ValueError: If `query` is empty or `top_k` is not positive.
            ProviderUnavailableError: If the model or store is unusable.
            EmbeddingError: If the query cannot be embedded.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        limit = top_k if top_k is not None else self.config.retrieval_top_k
        if limit <= 0:
            raise ValueError("top_k must be positive.")

        if not self.is_indexed():
            logger.warning(
                "Retrieval requested but the index is empty; returning no "
                "results. Build the index first with Indexer.build_index()."
            )
            return []

        embedding = self.embedder.embed_text(query)
        results = self.vector_db.search(embedding, top_k=limit, where=where)

        if min_score is not None:
            results = [hit for hit in results if hit.score >= min_score]

        # ChromaDB already returns nearest first; sorting explicitly
        # keeps the ordering guaranteed by this class rather than
        # inherited from a backend that could change.
        results.sort(key=lambda hit: hit.score, reverse=True)
        return results

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a query.

        The agent-facing entry point, returning the RetrievedChunk
        schema the scaffold was built against.

        Args:
            query: Natural language or code query.
            top_k: Maximum chunks to return. Defaults to
                `Config.retrieval_top_k`.
            where: Optional metadata filter.
            min_score: Optional similarity floor.

        Returns:
            Chunks ordered by relevance, highest first. Empty when the
            index is empty or nothing clears `min_score`.

        Raises:
            ValueError: If `query` is empty or `top_k` is not positive.
            ProviderUnavailableError: If the model or store is unusable.
            EmbeddingError: If the query cannot be embedded.
        """
        return [
            self._to_retrieved_chunk(hit)
            for hit in self.search(query, top_k=top_k, where=where,
                                   min_score=min_score)
        ]

    def retrieve_by_file(
        self,
        file_path: str,
        top_k: Optional[int] = None,
        query: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks belonging to one file.

        With a `query`, this is a similarity search scoped to the file.
        Without one there is nothing to rank against, so the file's
        chunks are returned in source order instead -- which is what a
        caller reading a file top to bottom actually wants.

        Args:
            file_path: File to retrieve chunks for, as recorded in the
                index.
            top_k: Maximum chunks to return. Defaults to
                `Config.retrieval_top_k`.
            query: Optional query to rank the file's chunks by.

        Returns:
            The file's chunks. Ranked by relevance when `query` is
            given, otherwise ordered by line number with a score of 0.0,
            since an unranked chunk has no meaningful similarity.

        Raises:
            ValueError: If `file_path` is empty or `top_k` is not
                positive.
            ProviderUnavailableError: If the store is unusable.
        """
        if not file_path or not str(file_path).strip():
            raise ValueError("file_path must be a non-empty string.")

        limit = top_k if top_k is not None else self.config.retrieval_top_k
        if limit <= 0:
            raise ValueError("top_k must be positive.")

        where = self.build_filter(file_path=file_path)

        if query is not None:
            return self.retrieve(query, top_k=limit, where=where)

        chunks = self.vector_db.list_chunks(where=where)
        chunks.sort(key=lambda chunk: chunk.line_start)
        return [
            self._to_retrieved_chunk(SearchResult(chunk=chunk, distance=0.0,
                                                  score=0.0))
            for chunk in chunks[:limit]
        ]

    def rerank(
        self, query: str, chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """
        Re-order retrieved chunks by relevance.

        Currently a stable sort on the similarity score already
        attached, which is a no-op for results straight out of
        `retrieve` and useful only when merging several result sets.
        It is deliberately not a second scoring pass: a cross-encoder
        would give better ordering but needs its own model, and
        pretending to re-rank without one would be worse than being
        explicit that this only sorts.

        Args:
            query: The original query. Unused today, kept because the
                cross-encoder implementation will need it and callers
                should not have to change.
            chunks: Chunks to re-order.

        Returns:
            The chunks ordered by score, highest first.
        """
        return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    @staticmethod
    def build_filter(
        file_path: Optional[str] = None,
        language: Optional[str] = None,
        kind: Optional[str] = None,
        class_name: Optional[str] = None,
        function_name: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build a metadata filter from named conditions.

        Spares callers from hand-writing ChromaDB filter syntax, which
        needs an explicit `$and` as soon as there is more than one
        condition -- a plain multi-key dict is rejected.

        The filterable keys mirror what VectorDB promotes to top-level
        metadata when storing a chunk.

        Args:
            file_path: Restrict to one file.
            language: Restrict to a language, e.g. "python".
            kind: Restrict to a chunk kind, e.g. "function", "method",
                "class".
            class_name: Restrict to one class.
            function_name: Restrict to one function or method.
            extra: Additional raw conditions merged in as-is.

        Returns:
            A filter suitable for `search(where=...)`, or None when no
            conditions were given.
        """
        conditions: Dict[str, Any] = {}
        if file_path:
            conditions["file_path"] = file_path
        if language:
            conditions["language"] = language
        if kind:
            conditions["kind"] = kind
        if class_name:
            conditions["class_name"] = class_name
        if function_name:
            conditions["function_name"] = function_name
        if extra:
            conditions.update(extra)

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions
        return {"$and": [{key: value} for key, value in conditions.items()]}

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_retrieved_chunk(hit: SearchResult) -> RetrievedChunk:
        """
        Flatten a SearchResult into the RetrievedChunk schema.

        RetrievedChunk carries only a source, content, score, and a
        metadata bag, so everything else the chunk knows is packed into
        that bag. The line range especially has to survive: it is what
        lets a citation be checked against the file, and a chunk that
        cannot be located is not evidence.

        The chunk's own metadata is nested rather than merged, so a key
        like `kind` cannot collide with a structural field.

        Args:
            hit: The search result to convert.

        Returns:
            The equivalent RetrievedChunk.
        """
        chunk = hit.chunk
        return RetrievedChunk(
            source=chunk.file_path,
            content=chunk.content,
            score=hit.score,
            metadata={
                "chunk_id": chunk.chunk_id,
                "file_path": chunk.file_path,
                "language": chunk.language,
                "class_name": chunk.class_name,
                "function_name": chunk.function_name,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "imports": list(chunk.imports),
                "distance": hit.distance,
                "chunk_metadata": dict(chunk.metadata),
            },
        )

    @staticmethod
    def to_code_chunks(results: Sequence[SearchResult]) -> List[CodeChunk]:
        """
        Extract the CodeChunk objects from a set of results.

        A convenience for the analysis layer, which works in CodeChunks
        and does not care about scores.

        Args:
            results: Results to unwrap.

        Returns:
            The underlying CodeChunk objects, order preserved.
        """
        return [hit.chunk for hit in results]
