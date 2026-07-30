"""
embeddings.py
=============

Turns CodeChunk objects into dense vectors using sentence-transformers.

The model named by `Config.embedding_model_name` is loaded once and
shared. Loading is the expensive part -- hundreds of megabytes of
weights and several seconds -- so the cache is keyed by model name at
module level rather than per instance: the Indexer, a notebook cell, and
a test can each construct their own EmbeddingGenerator without paying
for the model more than once.

Embeddings are returned as plain lists of floats, not tensors or numpy
arrays. That keeps this module free of any vector-database dependency:
ChromaDB, a JSON dump, or anything else can consume the output without
this file knowing which one it is.

Scope: embedding generation only. Storing vectors belongs to VectorDB,
and deciding what to embed belongs to the Chunker and Ingestor.

TODO: Move `DEFAULT_BATCH_SIZE` onto Config once a sensible
project-wide value is known from real ingestion runs.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Sequence

from ..config import Config
from ..exceptions.model_exceptions import EmbeddingError, ProviderUnavailableError
from ..schemas.schemas import CodeChunk

logger = logging.getLogger(__name__)

# Loaded models keyed by name, shared across every EmbeddingGenerator in
# the process. Guarded by a lock so two threads racing to embed cannot
# both pay to load the same weights.
_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


class EmbeddingGenerator:
    """
    Generates vector embeddings for CodeChunk objects.

    The model is loaded lazily on first use, so constructing an instance
    is cheap and a caller that never embeds anything never pays for the
    weights.
    """

    #: Chunks encoded per forward pass. Batching is what makes bulk
    #: embedding tractable -- encoding chunks one at a time wastes most
    #: of the model's throughput.
    DEFAULT_BATCH_SIZE: int = 32

    def __init__(
        self,
        config: Optional[Config] = None,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        model: Optional[Any] = None,
    ) -> None:
        """
        Initialize the EmbeddingGenerator.

        Args:
            config: Optional Config instance. A default is loaded when
                not supplied.
            model_name: Override for `Config.embedding_model_name`.
            batch_size: Override for the per-batch chunk count.
            normalize: When True, vectors are scaled to unit length.
                Left on by default because cosine similarity is then a
                plain dot product, and Euclidean distance -- what
                ChromaDB uses by default -- ranks identically to cosine.
                Without this, longer chunks get larger vectors and rank
                higher for reasons unrelated to relevance.
            model: A pre-loaded model, injected for tests. When given,
                nothing is loaded and the shared cache is bypassed. Must
                expose `encode()` and either `get_embedding_dimension()`
                or `get_sentence_embedding_dimension()`.

        Raises:
            ValueError: If `batch_size` is not positive.
        """
        self.config = config or Config.load()
        self.model_name = model_name or self.config.embedding_model_name
        self.normalize = normalize

        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE

        self._model = model

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> Any:
        """
        Return the embedding model, loading it on first use.

        Resolution order is instance, then the process-wide cache, then
        an actual load. The lock is checked-then-rechecked so a second
        thread arriving mid-load waits and reuses the result rather than
        loading a duplicate copy.

        Returns:
            The loaded SentenceTransformer instance.

        Raises:
            ProviderUnavailableError: If sentence-transformers is not
                installed, or the model cannot be loaded (typically a
                bad model name or no network on first download).
        """
        if self._model is not None:
            return self._model

        cached = _MODEL_CACHE.get(self.model_name)
        if cached is not None:
            self._model = cached
            return cached

        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(self.model_name)
            if cached is None:
                cached = self._load_sentence_transformer(self.model_name)
                _MODEL_CACHE[self.model_name] = cached
            self._model = cached

        return self._model

    def is_loaded(self) -> bool:
        """
        Report whether the model is ready without triggering a load.

        Args:
            None.

        Returns:
            True if this instance holds a model or the shared cache does.
        """
        return self._model is not None or self.model_name in _MODEL_CACHE

    @property
    def dimension(self) -> int:
        """
        Length of the vectors this generator produces.

        Loads the model if it is not already loaded, since the dimension
        is a property of the weights.

        Returns:
            The embedding dimension (768 for the default
            `all-mpnet-base-v2`).

        Raises:
            ProviderUnavailableError: If the model cannot be loaded.
            EmbeddingError: If the model does not report a dimension.
        """
        model = self.load_model()

        # sentence-transformers 5.x renamed this; the old name still
        # works but warns. Try the new name first so the project runs
        # clean on current versions without dropping support for older
        # ones, which requirements.txt still permits.
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension", None
        )
        if getter is None:
            raise EmbeddingError(
                f"Model {self.model_name!r} does not report an embedding "
                f"dimension."
            )
        size = getter()

        if not size:
            raise EmbeddingError(
                f"Model {self.model_name!r} reported an invalid embedding "
                f"dimension: {size!r}."
            )
        return int(size)

    @staticmethod
    def _load_sentence_transformer(model_name: str) -> Any:
        """
        Import sentence-transformers and construct the model.

        The import is deliberately deferred to call time rather than
        living at module scope. sentence-transformers pulls in torch and
        costs seconds to import, and the package as a whole should stay
        importable -- and the rest of the scaffold runnable -- on a
        machine where it was never installed.

        Args:
            model_name: Name of the sentence-transformers model.

        Returns:
            The constructed SentenceTransformer.

        Raises:
            ProviderUnavailableError: If the package is missing or the
                model cannot be constructed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ProviderUnavailableError(
                "sentence-transformers is not installed. Install it with "
                "`pip install sentence-transformers` to generate embeddings."
            ) from exc

        logger.info("Loading embedding model %r...", model_name)
        try:
            model = SentenceTransformer(model_name)
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Could not load embedding model {model_name!r}: {exc}"
            ) from exc

        logger.info("Embedding model %r ready.", model_name)
        return model

    # ------------------------------------------------------------------
    # Embedding text
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single string.

        Args:
            text: Text to embed.

        Returns:
            The embedding vector.

        Raises:
            ValueError: If `text` is empty or whitespace only, which
                would produce a vector carrying no meaning while still
                matching queries.
            ProviderUnavailableError: If the model cannot be loaded.
            EmbeddingError: If encoding fails.
        """
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text).__name__}.")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace only.")

        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Embed several strings in batches.

        Args:
            texts: Texts to embed.

        Returns:
            One vector per input, in the same order.

        Raises:
            ValueError: If any entry is not a non-empty string. Order is
                the contract here -- callers zip these back onto their
                own objects -- so a bad entry is rejected outright
                rather than skipped, which would silently misalign every
                vector after it.
            ProviderUnavailableError: If the model cannot be loaded.
            EmbeddingError: If encoding fails.
        """
        items = list(texts)
        if not items:
            return []

        for position, text in enumerate(items):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"texts[{position}] must be a non-empty string, got "
                    f"{text!r}."
                )

        model = self.load_model()
        try:
            vectors = model.encode(
                items,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed {len(items)} text(s) with "
                f"{self.model_name!r}: {exc}"
            ) from exc

        return self._to_float_lists(vectors, expected=len(items))

    # ------------------------------------------------------------------
    # Embedding chunks
    # ------------------------------------------------------------------

    def embed_chunk(self, chunk: CodeChunk) -> List[float]:
        """
        Embed a single CodeChunk.

        Args:
            chunk: The chunk to embed.

        Returns:
            The embedding vector.

        Raises:
            ValueError: If `chunk` is not a CodeChunk or has no content.
            ProviderUnavailableError: If the model cannot be loaded.
            EmbeddingError: If encoding fails.
        """
        return self.embed_chunks([chunk])[0]

    def embed_chunks(self, chunks: Sequence[CodeChunk]) -> List[List[float]]:
        """
        Embed several CodeChunks in batches.

        Args:
            chunks: The chunks to embed.

        Returns:
            One vector per chunk, in the same order, so a caller can zip
            them straight back onto the chunks they came from.

        Raises:
            ValueError: If any entry is not a CodeChunk or has no
                content.
            ProviderUnavailableError: If the model cannot be loaded.
            EmbeddingError: If encoding fails.
        """
        items = list(chunks)
        if not items:
            return []

        for position, chunk in enumerate(items):
            if not isinstance(chunk, CodeChunk):
                raise ValueError(
                    f"chunks[{position}] must be a CodeChunk, got "
                    f"{type(chunk).__name__}."
                )
            if not chunk.content.strip():
                raise ValueError(
                    f"chunks[{position}] ({chunk.chunk_id!r}) has no content "
                    f"to embed."
                )

        return self.embed_texts([self.build_embedding_text(c) for c in items])

    def build_embedding_text(self, chunk: CodeChunk) -> str:
        """
        Compose the text that actually gets embedded for a chunk.

        The chunk's raw content is not enough on its own. A method body
        rarely repeats its own qualified name, never mentions the file
        it lives in, and refers to imported names that appear nowhere
        inside it -- so a query like "how does the tool registry
        dispatch calls" has little to match against. Prefixing the
        location, the qualified name, and the imports puts those terms
        into the vector.

        `chunk.content` itself is left untouched. The enrichment exists
        only in the embedded text, so the stored content remains a
        byte-exact slice of the source and grounding still works.

        Args:
            chunk: The chunk to describe.

        Returns:
            The text to embed.
        """
        header: List[str] = [f"File: {chunk.file_path}"]

        qualified = ".".join(
            part for part in (chunk.class_name, chunk.function_name) if part
        )
        if qualified:
            header.append(f"Name: {qualified}")

        heading = chunk.metadata.get("heading")
        if heading:
            header.append(f"Section: {heading}")

        if chunk.imports:
            header.append("Imports: " + "; ".join(chunk.imports))

        return "\n".join(header) + "\n\n" + chunk.content

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float_lists(vectors: Any, expected: int) -> List[List[float]]:
        """
        Convert a model's output into plain lists of floats.

        Keeping the boundary at plain Python types is what lets VectorDB
        stay swappable: nothing downstream has to handle a torch tensor
        or a numpy array.

        Args:
            vectors: Whatever `model.encode()` returned.
            expected: Number of vectors the caller asked for.

        Returns:
            One list of floats per input.

        Raises:
            EmbeddingError: If the output cannot be converted, or holds
                a different number of vectors than requested -- which
                would silently misalign vectors against their chunks.
        """
        try:
            converted = [
                [float(value) for value in vector] for vector in vectors
            ]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError(
                f"Embedding model returned an unusable result: {exc}"
            ) from exc

        if len(converted) != expected:
            raise EmbeddingError(
                f"Embedding model returned {len(converted)} vectors for "
                f"{expected} input(s)."
            )
        return converted
