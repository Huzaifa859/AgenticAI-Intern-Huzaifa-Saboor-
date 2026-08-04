"""
vectordb.py
===========

Persistence for embedded CodeChunks, backed by ChromaDB.

This layer stores and retrieves; it does not decide what to store or how
to use what comes back. Embedding a query, choosing `top_k`,
re-ranking, and assembling context all belong to the Retriever, so the
only thing this module knows about a query is the vector it was handed.

`chunk_id` is the primary key throughout. Writes go through `upsert`
rather than `add`, which is what makes re-indexing a repository safe:
the same chunk written twice updates in place instead of raising or
quietly creating a duplicate that would then compete with itself in
search results.

Search results come back as CodeChunk objects rather than raw ChromaDB
payloads, so the Retriever works with the same type the Chunker
produced and nothing downstream has to understand Chroma's response
shape.

TODO: The exception hierarchy has no persistence category, so storage
failures currently reuse `ProviderUnavailableError` from the model
layer. Add a `StorageError` to `exceptions/` and switch to it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import Config
from ..exceptions.model_exceptions import ProviderUnavailableError
from ..schemas.schemas import CodeChunk

logger = logging.getLogger(__name__)

# Metadata keys this module owns. A chunk's own metadata cannot use
# them, or a promoted filter key would overwrite a structural field.
_RESERVED_METADATA_KEYS = frozenset(
    {
        "file_path",
        "language",
        "class_name",
        "function_name",
        "line_start",
        "line_end",
        "imports_json",
        "extra_json",
    }
)


@dataclass
class SearchResult:
    """
    One hit from a similarity search.

    Attributes:
        chunk: The stored chunk, reconstructed from the collection.
        distance: Raw distance reported by ChromaDB. Lower is closer.
        score: Similarity in the range [-1, 1], higher being better.
            Derived here rather than by the caller because the
            conversion depends on the collection's distance metric,
            which is this layer's business to know.
    """

    chunk: CodeChunk
    distance: float
    score: float


class VectorDB:
    """
    Persistent ChromaDB collection holding embedded CodeChunks.

    The client and collection are opened lazily on first use, so
    constructing an instance touches no disk and a caller that never
    queries never pays for the import.
    """

    #: Records per write call. ChromaDB enforces its own per-request
    #: ceiling, and a whole-repository index can exceed it in one go.
    DEFAULT_BATCH_SIZE: int = 256

    #: Distance metric for new collections. Cosine pairs with the
    #: unit-length vectors EmbeddingGenerator produces and makes
    #: `score` a plain cosine similarity.
    DISTANCE_METRIC: str = "cosine"

    def __init__(
        self,
        config: Optional[Config] = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        batch_size: Optional[int] = None,
        client: Optional[Any] = None,
    ) -> None:
        """
        Initialize the VectorDB.

        Args:
            config: Optional Config instance. A default is loaded when
                not supplied.
            collection_name: Override for
                `Config.chroma_collection_name`.
            persist_directory: Override for
                `Config.chroma_persist_directory`.
            batch_size: Override for the per-write record count.
            client: A pre-built ChromaDB client, injected for tests so a
                caller can supply an in-memory client.

        Raises:
            ValueError: If `batch_size` is not positive.
        """
        self.config = config or Config.load()
        self.collection_name = collection_name or self.config.chroma_collection_name
        self.persist_directory = (
            persist_directory or self.config.chroma_persist_directory
        )

        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE

        self._client = client
        self._collection: Optional[Any] = None

    # ------------------------------------------------------------------
    # Client and collection lifecycle
    # ------------------------------------------------------------------

    def get_client(self) -> Any:
        """
        Return the ChromaDB client, creating it on first use.

        Returns:
            The persistent ChromaDB client.

        Raises:
            ProviderUnavailableError: If chromadb is not installed or
                the persistence directory cannot be opened.
        """
        if self._client is not None:
            return self._client

        chromadb, settings_cls = self._import_chromadb()

        try:
            Path(self.persist_directory).expanduser().mkdir(
                parents=True, exist_ok=True
            )
        except OSError as exc:
            raise ProviderUnavailableError(
                f"Could not create the vector store directory "
                f"{self.persist_directory!r}: {exc}"
            ) from exc

        try:
            self._client = chromadb.PersistentClient(
                path=str(Path(self.persist_directory).expanduser()),
                settings=settings_cls(anonymized_telemetry=False),
            )
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Could not open the ChromaDB store at "
                f"{self.persist_directory!r}: {exc}"
            ) from exc

        logger.info("ChromaDB store ready at %s", self.persist_directory)
        return self._client

    def get_collection(self) -> Any:
        """
        Return the collection, creating it if it does not exist.

        Args:
            None.

        Returns:
            The ChromaDB collection named by Config.

        Raises:
            ProviderUnavailableError: If the collection cannot be opened
                or created.
        """
        if self._collection is not None:
            return self._collection

        client = self.get_client()
        try:
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.DISTANCE_METRIC},
            )
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Could not open collection {self.collection_name!r}: {exc}"
            ) from exc

        return self._collection

    def _import_chromadb(self) -> Any:
        """
        Import chromadb at call time.

        Deferred rather than imported at module scope so the package
        stays importable, and the rest of the scaffold runnable, on a
        machine where chromadb was never installed.

        Args:
            None.

        Returns:
            A tuple of (chromadb module, Settings class).

        Raises:
            ProviderUnavailableError: If chromadb is not installed.
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise ProviderUnavailableError(
                "chromadb is not installed. Install it with "
                "`pip install chromadb` to use the vector store."
            ) from exc
        return chromadb, Settings

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_chunk(self, chunk: CodeChunk, embedding: Sequence[float]) -> str:
        """
        Store or replace a single chunk.

        Args:
            chunk: The chunk to store.
            embedding: Its embedding vector.

        Returns:
            The chunk's ID.

        Raises:
            ValueError: If the chunk or embedding is invalid.
            ProviderUnavailableError: If the write fails.
        """
        self.add_chunks([chunk], [embedding])
        return chunk.chunk_id

    def add_chunks(
        self,
        chunks: Sequence[CodeChunk],
        embeddings: Sequence[Sequence[float]],
        batch_size: Optional[int] = None,
    ) -> int:
        """
        Store or replace many chunks in batches.

        Uses upsert, so writing a chunk whose ID already exists replaces
        it. Re-indexing a repository is therefore idempotent instead of
        accumulating stale copies of every chunk that moved.

        Duplicate IDs *within a single call* are collapsed before the
        write, keeping the last occurrence. ChromaDB rejects a batch
        containing repeated IDs outright, which would otherwise fail the
        entire index over one repeated name.

        Args:
            chunks: Chunks to store.
            embeddings: One embedding per chunk, in the same order.
            batch_size: Override for the per-write record count.

        Returns:
            The number of records written after collapsing duplicates.

        Raises:
            ValueError: If the inputs are misaligned or invalid.
            ProviderUnavailableError: If the write fails.
        """
        items = list(chunks)
        vectors = [list(vector) for vector in embeddings]

        if not items:
            return 0
        if len(items) != len(vectors):
            raise ValueError(
                f"chunks and embeddings must be the same length, got "
                f"{len(items)} and {len(vectors)}."
            )

        self._validate_chunks(items)
        self._validate_embeddings(vectors)

        ids, documents, metadatas, deduped = self._collapse_duplicates(items, vectors)

        collection = self.get_collection()
        size = batch_size or self.batch_size
        for start in range(0, len(ids), size):
            stop = start + size
            try:
                collection.upsert(
                    ids=ids[start:stop],
                    embeddings=deduped[start:stop],
                    documents=documents[start:stop],
                    metadatas=metadatas[start:stop],
                )
            except Exception as exc:
                raise ProviderUnavailableError(
                    f"Failed to write {len(ids[start:stop])} chunk(s) to "
                    f"collection {self.collection_name!r}: {exc}"
                ) from exc

        return len(ids)

    def update_chunk(self, chunk: CodeChunk, embedding: Sequence[float]) -> str:
        """
        Replace an existing chunk.

        An alias for `add_chunk`, present because "update" is what the
        caller means when re-indexing a changed file. Both are upserts,
        so neither fails when the ID is absent.

        Args:
            chunk: The chunk to store.
            embedding: Its embedding vector.

        Returns:
            The chunk's ID.

        Raises:
            ValueError: If the chunk or embedding is invalid.
            ProviderUnavailableError: If the write fails.
        """
        return self.add_chunk(chunk, embedding)

    # ------------------------------------------------------------------
    # Deleting
    # ------------------------------------------------------------------

    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a single chunk by ID.

        Args:
            chunk_id: ID of the chunk to remove.

        Returns:
            True if a chunk was removed, False if the ID was not
            present. ChromaDB does not report this, so presence is
            checked first.

        Raises:
            ValueError: If `chunk_id` is empty.
            ProviderUnavailableError: If the delete fails.
        """
        return self.delete_chunks([chunk_id]) > 0

    def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        """
        Delete several chunks by ID.

        Args:
            chunk_ids: IDs of the chunks to remove. IDs that are not
                present are ignored.

        Returns:
            The number of chunks actually removed.

        Raises:
            ValueError: If any ID is empty.
            ProviderUnavailableError: If the delete fails.
        """
        ids = list(chunk_ids)
        if not ids:
            return 0

        for position, chunk_id in enumerate(ids):
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    f"chunk_ids[{position}] must be a non-empty string, got "
                    f"{chunk_id!r}."
                )

        present = self._existing_ids(ids)
        if not present:
            return 0

        collection = self.get_collection()
        try:
            collection.delete(ids=present)
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Failed to delete {len(present)} chunk(s) from collection "
                f"{self.collection_name!r}: {exc}"
            ) from exc

        return len(present)

    def delete_by_metadata(self, where: Dict[str, Any]) -> int:
        """
        Delete every chunk matching a metadata filter.

        The path for re-indexing a single file: delete everything
        recorded for it, then write the new chunks.

        Args:
            where: A ChromaDB metadata filter, e.g.
                `{"file_path": "src/app.py"}`.

        Returns:
            The number of chunks removed.

        Raises:
            ValueError: If `where` is empty.
            ProviderUnavailableError: If the delete fails.
        """
        if not where:
            raise ValueError(
                "where must be a non-empty filter. Use delete_all() to clear "
                "the collection."
            )

        collection = self.get_collection()
        try:
            matched = collection.get(where=where, include=[])
            ids = list(matched.get("ids") or [])
            if ids:
                collection.delete(ids=ids)
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Failed to delete by filter {where!r}: {exc}"
            ) from exc

        return len(ids)

    def delete_all(self) -> int:
        """
        Remove every chunk from the collection.

        Implemented by dropping and recreating the collection rather
        than issuing a filterless delete, which recent ChromaDB versions
        reject. Recreating also discards the HNSW index, so a rebuilt
        collection carries nothing over from the old one.

        Args:
            None.

        Returns:
            The number of chunks that were present before clearing.

        Raises:
            ProviderUnavailableError: If the collection cannot be
                recreated.
        """
        previous = self.count()
        client = self.get_client()

        try:
            client.delete_collection(name=self.collection_name)
        except Exception as exc:
            logger.debug("Collection %r not dropped: %s", self.collection_name, exc)

        self._collection = None
        try:
            self.get_collection()
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Could not recreate collection {self.collection_name!r}: {exc}"
            ) from exc

        logger.info("Cleared %d chunk(s) from %r", previous, self.collection_name)
        return previous

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def search(
        self,
        embedding: Sequence[float],
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Find the chunks nearest to a query embedding.

        Embedding the query is the Retriever's job; this method only
        knows the vector it was given.

        Args:
            embedding: The query vector.
            top_k: Maximum hits to return. Defaults to
                `Config.retrieval_top_k`.
            where: Optional ChromaDB metadata filter, e.g.
                `{"language": "python"}` or
                `{"file_path": "src/app.py"}`. Applied inside the search
                so the filter narrows the candidate set rather than
                trimming results afterwards, which would return fewer
                than `top_k` hits.

        Returns:
            Hits ordered nearest first.

        Raises:
            ValueError: If the embedding is invalid or `top_k` is not
                positive.
            ProviderUnavailableError: If the query fails.
        """
        vector = list(embedding)
        self._validate_embeddings([vector])

        limit = top_k if top_k is not None else self.config.retrieval_top_k
        if limit <= 0:
            raise ValueError("top_k must be positive.")

        collection = self.get_collection()
        try:
            response = collection.query(
                query_embeddings=[vector],
                n_results=limit,
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Similarity search failed on collection "
                f"{self.collection_name!r}: {exc}"
            ) from exc

        return self._to_search_results(response)

    def get_chunk(self, chunk_id: str) -> Optional[CodeChunk]:
        """
        Fetch a single chunk by ID.

        Args:
            chunk_id: ID of the chunk to fetch.

        Returns:
            The chunk, or None if the ID is not present.

        Raises:
            ValueError: If `chunk_id` is empty.
            ProviderUnavailableError: If the read fails.
        """
        if not chunk_id or not str(chunk_id).strip():
            raise ValueError("chunk_id must be a non-empty string.")

        results = self.get_chunks([chunk_id])
        return results[0] if results else None

    def get_chunks(self, chunk_ids: Sequence[str]) -> List[CodeChunk]:
        """
        Fetch several chunks by ID.

        Args:
            chunk_ids: IDs to fetch. Missing IDs are skipped rather than
                raising.

        Returns:
            The chunks that were found.

        Raises:
            ProviderUnavailableError: If the read fails.
        """
        ids = [str(chunk_id) for chunk_id in chunk_ids]
        if not ids:
            return []

        collection = self.get_collection()
        try:
            response = collection.get(
                ids=ids, include=["documents", "metadatas"]
            )
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Failed to read {len(ids)} chunk(s) from collection "
                f"{self.collection_name!r}: {exc}"
            ) from exc

        return self._to_chunks(response)

    def has_chunk(self, chunk_id: str) -> bool:
        """
        Report whether a chunk ID is stored.

        Args:
            chunk_id: ID to test.

        Returns:
            True if the ID is present.

        Raises:
            ProviderUnavailableError: If the read fails.
        """
        return bool(self._existing_ids([chunk_id]))

    def list_chunks(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[CodeChunk]:
        """
        List chunks by metadata filter, without a similarity search.

        Backs file-scoped lookups such as "every chunk from src/app.py",
        where there is no query vector to search with.

        Args:
            where: Optional ChromaDB metadata filter. Omit to list
                everything.
            limit: Maximum chunks to return.

        Returns:
            The matching chunks.

        Raises:
            ValueError: If `limit` is not positive.
            ProviderUnavailableError: If the read fails.
        """
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive.")

        collection = self.get_collection()
        try:
            response = collection.get(
                where=where or None,
                limit=limit,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Failed to list chunks from collection "
                f"{self.collection_name!r}: {exc}"
            ) from exc

        return self._to_chunks(response)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def count(self) -> int:
        """
        Count the chunks in the collection.

        Args:
            None.

        Returns:
            The number of stored chunks.

        Raises:
            ProviderUnavailableError: If the count fails.
        """
        collection = self.get_collection()
        try:
            return int(collection.count())
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Could not count collection {self.collection_name!r}: {exc}"
            ) from exc

    def stats(self) -> Dict[str, Any]:
        """
        Summarize the collection's contents.

        Reads every record's metadata to build the breakdowns, which is
        acceptable at the proposal's ceiling of 100 files but is not a
        hot-path call.

        Args:
            None.

        Returns:
            The collection name, storage path, distance metric, chunk
            count, distinct file count, and per-language and per-kind
            breakdowns.

        Raises:
            ProviderUnavailableError: If the read fails.
        """
        collection = self.get_collection()
        total = self.count()

        summary: Dict[str, Any] = {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "distance_metric": self.DISTANCE_METRIC,
            "chunk_count": total,
            "file_count": 0,
            "languages": {},
            "kinds": {},
        }
        if total == 0:
            return summary

        try:
            response = collection.get(include=["metadatas"])
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Could not read metadata from collection "
                f"{self.collection_name!r}: {exc}"
            ) from exc

        files = set()
        languages: Dict[str, int] = {}
        kinds: Dict[str, int] = {}
        for metadata in response.get("metadatas") or []:
            metadata = metadata or {}
            if metadata.get("file_path"):
                files.add(metadata["file_path"])
            language = str(metadata.get("language", "unknown"))
            languages[language] = languages.get(language, 0) + 1
            kind = str(metadata.get("kind", metadata.get("chunk_strategy", "unknown")))
            kinds[kind] = kinds.get(kind, 0) + 1

        summary["file_count"] = len(files)
        summary["languages"] = dict(sorted(languages.items()))
        summary["kinds"] = dict(sorted(kinds.items()))
        return summary

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_chunks(chunks: Sequence[CodeChunk]) -> None:
        """
        Reject anything that is not a storable CodeChunk.

        Args:
            chunks: Chunks to check.

        Raises:
            ValueError: If an entry is not a CodeChunk or has no ID.
        """
        for position, chunk in enumerate(chunks):
            if not isinstance(chunk, CodeChunk):
                raise ValueError(
                    f"chunks[{position}] must be a CodeChunk, got "
                    f"{type(chunk).__name__}."
                )
            if not chunk.chunk_id or not chunk.chunk_id.strip():
                raise ValueError(
                    f"chunks[{position}] has an empty chunk_id; it is the "
                    f"primary key and cannot be blank."
                )

    @staticmethod
    def _validate_embeddings(embeddings: Sequence[Sequence[float]]) -> None:
        """
        Reject malformed or inconsistently sized embeddings.

        A mixed-dimension batch is caught here rather than at the
        ChromaDB boundary, where the error names neither the offending
        position nor the expected size.

        Args:
            embeddings: Vectors to check.

        Raises:
            ValueError: If a vector is empty, non-numeric, or a
                different length from the first.
        """
        expected: Optional[int] = None
        for position, vector in enumerate(embeddings):
            if not isinstance(vector, (list, tuple)) or not vector:
                raise ValueError(
                    f"embeddings[{position}] must be a non-empty sequence of "
                    f"numbers."
                )
            if not all(isinstance(value, (int, float)) for value in vector):
                raise ValueError(
                    f"embeddings[{position}] must contain only numbers."
                )
            if expected is None:
                expected = len(vector)
            elif len(vector) != expected:
                raise ValueError(
                    f"embeddings[{position}] has dimension {len(vector)}, "
                    f"expected {expected}."
                )

    # ------------------------------------------------------------------
    # Record conversion
    # ------------------------------------------------------------------

    def _collapse_duplicates(
        self,
        chunks: Sequence[CodeChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> Any:
        """
        Collapse repeated IDs within one write, keeping the last.

        Args:
            chunks: Chunks about to be written.
            embeddings: Their embeddings, in the same order.

        Returns:
            A tuple of (ids, documents, metadatas, embeddings), aligned
            and free of repeated IDs.
        """
        by_id: Dict[str, Any] = {}
        for chunk, vector in zip(chunks, embeddings):
            if chunk.chunk_id in by_id:
                logger.warning(
                    "Duplicate chunk_id %r in one write; keeping the last.",
                    chunk.chunk_id,
                )
            by_id[chunk.chunk_id] = (chunk, list(vector))

        ids = list(by_id)
        documents = [by_id[key][0].content for key in ids]
        metadatas = [self._to_metadata(by_id[key][0]) for key in ids]
        vectors = [by_id[key][1] for key in ids]
        return ids, documents, metadatas, vectors

    @staticmethod
    def _to_metadata(chunk: CodeChunk) -> Dict[str, Any]:
        """
        Flatten a CodeChunk into ChromaDB-compatible metadata.

        ChromaDB only accepts scalar metadata values, so the two
        structured fields are JSON-encoded: `imports` and the chunk's
        own `metadata` dict. Scalar entries from that dict are *also*
        copied up as top-level keys, because JSON strings are not
        filterable and filtering on `kind` or `chunk_strategy` is the
        whole point of storing them. The JSON copy stays authoritative
        on read, so the redundancy cannot drift.

        Args:
            chunk: The chunk to flatten.

        Returns:
            A metadata mapping ChromaDB will accept.
        """
        metadata: Dict[str, Any] = {
            "file_path": chunk.file_path,
            "language": chunk.language,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "imports_json": json.dumps(chunk.imports),
            "extra_json": json.dumps(chunk.metadata, default=str),
        }

        # Omitted rather than set to None: ChromaDB rejects null values.
        if chunk.class_name:
            metadata["class_name"] = chunk.class_name
        if chunk.function_name:
            metadata["function_name"] = chunk.function_name

        for key, value in chunk.metadata.items():
            if key in _RESERVED_METADATA_KEYS:
                continue
            if isinstance(value, (bool, int, float, str)):
                metadata[key] = value

        return metadata

    @staticmethod
    def _to_chunk(
        chunk_id: str, document: str, metadata: Dict[str, Any]
    ) -> CodeChunk:
        """
        Rebuild a CodeChunk from a stored record.

        Args:
            chunk_id: The record's ID.
            document: The stored content.
            metadata: The stored metadata.

        Returns:
            The reconstructed chunk.

        Raises:
            ProviderUnavailableError: If the record is missing the
                structural fields a CodeChunk needs.
        """
        metadata = metadata or {}

        def decode(key: str, fallback: Any) -> Any:
            """Decode a JSON metadata field, tolerating corruption."""
            raw = metadata.get(key)
            if not raw:
                return fallback
            try:
                return json.loads(raw)
            except (TypeError, ValueError):
                logger.warning("Could not decode %s for chunk %r.", key, chunk_id)
                return fallback

        try:
            return CodeChunk(
                chunk_id=chunk_id,
                file_path=str(metadata["file_path"]),
                language=str(metadata["language"]),
                class_name=metadata.get("class_name"),
                function_name=metadata.get("function_name"),
                line_start=int(metadata["line_start"]),
                line_end=int(metadata["line_end"]),
                imports=decode("imports_json", []),
                content=document or "",
                metadata=decode("extra_json", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderUnavailableError(
                f"Stored record {chunk_id!r} is missing fields required to "
                f"rebuild a CodeChunk: {exc}"
            ) from exc

    def _to_chunks(self, response: Dict[str, Any]) -> List[CodeChunk]:
        """
        Convert a ChromaDB `get` response into CodeChunks.

        Args:
            response: The raw response.

        Returns:
            The reconstructed chunks, in the order returned.
        """
        ids = response.get("ids") or []
        documents = response.get("documents") or []
        metadatas = response.get("metadatas") or []

        chunks: List[CodeChunk] = []
        for position, chunk_id in enumerate(ids):
            document = documents[position] if position < len(documents) else ""
            metadata = metadatas[position] if position < len(metadatas) else {}
            chunks.append(self._to_chunk(chunk_id, document, metadata))
        return chunks

    def _to_search_results(self, response: Dict[str, Any]) -> List[SearchResult]:
        """
        Convert a ChromaDB `query` response into SearchResults.

        ChromaDB nests its query response one level per query embedding.
        Only one is ever sent, so the first row is unwrapped here rather
        than leaking the nesting to the Retriever.

        Args:
            response: The raw response.

        Returns:
            Hits ordered nearest first.
        """
        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        results: List[SearchResult] = []
        for position, chunk_id in enumerate(ids):
            document = documents[position] if position < len(documents) else ""
            metadata = metadatas[position] if position < len(metadatas) else {}
            distance = (
                float(distances[position]) if position < len(distances) else 0.0
            )
            results.append(
                SearchResult(
                    chunk=self._to_chunk(chunk_id, document, metadata),
                    distance=distance,
                    # Cosine distance is 1 - similarity, so this inverts
                    # back to a similarity the Retriever can threshold.
                    score=1.0 - distance,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _existing_ids(self, chunk_ids: Sequence[str]) -> List[str]:
        """
        Filter a list of IDs down to those actually stored.

        ChromaDB's delete is silent about what it removed, so presence
        is established first to give callers an honest return value.

        Args:
            chunk_ids: IDs to check.

        Returns:
            The subset that exists in the collection.

        Raises:
            ProviderUnavailableError: If the read fails.
        """
        ids = [str(chunk_id) for chunk_id in chunk_ids]
        if not ids:
            return []

        collection = self.get_collection()
        try:
            response = collection.get(ids=ids, include=[])
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Could not look up {len(ids)} chunk ID(s) in collection "
                f"{self.collection_name!r}: {exc}"
            ) from exc

        return list(response.get("ids") or [])
