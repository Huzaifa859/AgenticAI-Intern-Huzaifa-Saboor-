"""
chunker.py
==========

Splits repository files into CodeChunk objects for embedding.

Python files are chunked by AST so each chunk is a semantically whole
unit -- one per function, one per class -- rather than an arbitrary
window of N lines. That matters downstream: a chunk that starts halfway
through a function embeds poorly and, worse, gives the grounding checker
a line range that does not correspond to anything a developer would
recognize.

Markdown and plain text have no AST, so they are split on structure
instead: Markdown at its headings, plain text at paragraph boundaries.

Chunk text is always an exact slice of the source. Nothing is
reformatted, re-indented, or whitespace-normalized, because
GroundingChecker verifies a quoted snippet by re-reading the file at the
recorded line range and comparing. Any cosmetic rewrite here turns into
a false "unverified" verdict there.

Scope: chunk creation only. Deciding *which* files to feed in belongs to
the Ingestor; embedding and storage belong to EmbeddingGenerator and
VectorDB.

TODO: Move `DEFAULT_MAX_CHUNK_LINES` onto Config once the ingestion
pipeline is tuned and a sensible project-wide value is known.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import Config
from ..exceptions.tool_exceptions import UnsupportedFileTypeError
from ..schemas.schemas import CodeChunk
from ..tools.filesystem_tools import FilesystemTools

logger = logging.getLogger(__name__)

# Matches an ATX Markdown heading ("## Title"). Up to three leading
# spaces are allowed, matching the CommonMark rule.
_HEADING_PATTERN = re.compile(r"^ {0,3}(#{1,6})\s+(.*)$")

# Matches a fenced code block delimiter (``` or ~~~).
_FENCE_PATTERN = re.compile(r"^ {0,3}(`{3,}|~{3,})")


class Chunker:
    """
    Turns file contents into CodeChunk objects.

    Stateless between calls: every method takes the content it operates
    on, so one instance can be reused across a whole repository.
    """

    #: Extensions this chunker handles, mapped to the `language` value
    #: recorded on each chunk. Anything absent is out of scope.
    SUPPORTED_LANGUAGES: Dict[str, str] = {
        ".py": "python",
        ".md": "markdown",
        ".txt": "text",
    }

    #: Ceiling on lines per chunk for text-structured content. Only
    #: applies to Markdown/text and to the syntax-error fallback; AST
    #: chunks follow the code's own boundaries regardless of length,
    #: since splitting a function in half defeats the purpose.
    DEFAULT_MAX_CHUNK_LINES: int = 80

    def __init__(
        self,
        config: Optional[Config] = None,
        max_chunk_lines: Optional[int] = None,
        filesystem: Optional[FilesystemTools] = None,
    ) -> None:
        """
        Initialize the Chunker.

        Args:
            config: Optional Config instance. A default is loaded when
                not supplied.
            max_chunk_lines: Override for the per-chunk line ceiling
                applied to text-structured content.
            filesystem: Optional FilesystemTools used by `chunk_file`.
                Constructed lazily from `config` when omitted, so a
                caller that only ever uses `chunk` never pays for
                workspace validation.

        Raises:
            ValueError: If `max_chunk_lines` is not positive.
        """
        self.config = config or Config.load()

        if max_chunk_lines is not None and max_chunk_lines <= 0:
            raise ValueError("max_chunk_lines must be positive.")
        self.max_chunk_lines = max_chunk_lines or self.DEFAULT_MAX_CHUNK_LINES

        self._filesystem = filesystem

    # ------------------------------------------------------------------
    # File type support
    # ------------------------------------------------------------------

    def language_for(self, file_path: str) -> Optional[str]:
        """
        Determine the chunker language for a path.

        Args:
            file_path: Path to inspect. Only the extension is read; the
                file need not exist.

        Returns:
            The language name, or None if the extension is out of scope.
        """
        return self.SUPPORTED_LANGUAGES.get(Path(file_path).suffix.lower())

    def supports(self, file_path: str) -> bool:
        """
        Report whether a path is a file type this chunker can handle.

        The non-raising counterpart to `chunk`, meant for the Ingestor:
        it filters a repository walk with this and skips the rest, so
        unsupported files are ignored during a bulk walk while a direct
        request to chunk one still fails loudly.

        Args:
            file_path: Path to test.

        Returns:
            True if the extension is supported.
        """
        return self.language_for(file_path) is not None

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def chunk_file(self, file_path: str) -> List[CodeChunk]:
        """
        Read a file from the workspace and chunk it.

        Reading is delegated to FilesystemTools so the sandbox check and
        the size ceiling from Config apply here exactly as they do
        everywhere else, rather than being restated.

        Args:
            file_path: Path to the file, relative to the workspace root.

        Returns:
            The file's chunks, in source order. Empty if the file has no
            substantive content.

        Raises:
            UnsupportedFileTypeError: If the extension is out of scope.
            PathOutsideWorkspaceError: If the path escapes the workspace.
            FileTooLargeError: If the file exceeds the size ceiling.
            ToolExecutionError: If the file is missing or unreadable.
        """
        self._require_supported(file_path)
        content = self._require_filesystem().read_file(file_path)
        return self.chunk(content, file_path)

    def chunk(self, content: str, file_path: str) -> List[CodeChunk]:
        """
        Split file content into chunks.

        `file_path` is required rather than optional because it decides
        the chunking strategy and is recorded on every chunk; without it
        a chunk cannot be traced back to source, which is what makes it
        useful for grounding.

        Args:
            content: The file's full text.
            file_path: Path the content came from, ideally relative to
                the repository root so the index stays valid no matter
                where the repository was cloned.

        Returns:
            The content's chunks, in source order. Empty if the content
            is blank.

        Raises:
            ValueError: If `file_path` is empty.
            UnsupportedFileTypeError: If the extension is out of scope.
        """
        language = self._require_supported(file_path)

        if not content or not content.strip():
            return []

        if language == "python":
            chunks = self._chunk_python(content, file_path)
        else:
            chunks = self._chunk_structured_text(content, file_path, language)

        return self._ensure_unique_ids(chunks)

    # ------------------------------------------------------------------
    # Python chunking
    # ------------------------------------------------------------------

    def _chunk_python(self, content: str, file_path: str) -> List[CodeChunk]:
        """
        Chunk Python source using its AST.

        Falls back to text chunking when the file will not parse. A
        repository with one broken file should still be fully indexable
        -- and a file that fails to parse is often exactly the file a
        user wants to ask about.

        Args:
            content: Python source text.
            file_path: Path recorded on each chunk.

        Returns:
            One chunk per function and per class, in source order.
        """
        lines = content.split("\n")

        try:
            tree = ast.parse(content)
        except (SyntaxError, ValueError) as exc:
            logger.warning(
                "AST parse failed for %s (%s); falling back to text chunking.",
                file_path,
                exc,
            )
            return self._chunk_structured_text(
                content,
                file_path,
                language="python",
                strategy="text_fallback",
                extra_metadata={"parse_error": str(exc)},
            )

        imports = self._collect_imports(tree, lines)

        chunks: List[CodeChunk] = []
        for node in tree.body:
            chunks.extend(
                self._chunk_node(node, lines, file_path, imports, class_path=())
            )

        if not chunks:
            # A module of pure top-level statements (a script, a
            # constants file). Indexing it as text beats losing it.
            logger.debug(
                "No functions or classes found in %s; chunking as text.", file_path
            )
            return self._chunk_structured_text(
                content,
                file_path,
                language="python",
                strategy="python_module_text",
            )

        return chunks

    def _chunk_node(
        self,
        node: ast.AST,
        lines: List[str],
        file_path: str,
        imports: List[str],
        class_path: Tuple[str, ...],
    ) -> List[CodeChunk]:
        """
        Produce chunks for a single AST node, recursing into classes.

        Args:
            node: The AST node to chunk.
            lines: The file's lines, used to slice exact source text.
            file_path: Path recorded on each chunk.
            imports: The file's import statements.
            class_path: Names of the enclosing classes, outermost first.
                Empty at module level.

        Returns:
            Chunks for this node and anything nested inside it. Empty
            for nodes that are neither functions nor classes.
        """
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return [
                self._build_python_chunk(
                    lines=lines,
                    file_path=file_path,
                    imports=imports,
                    name=node.name,
                    class_path=class_path,
                    line_start=self._node_start(node),
                    line_end=node.end_lineno or node.lineno,
                    kind="method" if class_path else "function",
                    decorators=self._decorator_names(node),
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                )
            ]

        if isinstance(node, ast.ClassDef):
            return self._chunk_class(node, lines, file_path, imports, class_path)

        return []

    def _chunk_class(
        self,
        node: ast.ClassDef,
        lines: List[str],
        file_path: str,
        imports: List[str],
        class_path: Tuple[str, ...],
    ) -> List[CodeChunk]:
        """
        Chunk a class and its members.

        The class chunk covers the declaration, docstring, and
        class-level attributes, stopping where the first member starts.
        Members then get their own chunks. The alternative -- a class
        chunk holding the entire class *plus* separate method chunks --
        would store every method's text twice, and a retrieval returning
        both the class and one of its methods would spend two of its
        `retrieval_top_k` slots on the same code.

        Args:
            node: The ClassDef node.
            lines: The file's lines, used to slice exact source text.
            file_path: Path recorded on each chunk.
            imports: The file's import statements.
            class_path: Names of the enclosing classes, outermost first.

        Returns:
            The class chunk followed by its members' chunks.
        """
        class_start = self._node_start(node)
        class_end = node.end_lineno or class_start
        nested_path = class_path + (node.name,)

        member_chunks: List[CodeChunk] = []
        member_starts: List[int] = []
        for member in node.body:
            if isinstance(
                member, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                member_starts.append(self._node_start(member))
                member_chunks.extend(
                    self._chunk_node(member, lines, file_path, imports, nested_path)
                )

        # Stop the header chunk before the first member, but never
        # before the class statement itself.
        header_end = min(member_starts) - 1 if member_starts else class_end
        header_end = max(class_start, header_end)

        class_chunk = self._build_python_chunk(
            lines=lines,
            file_path=file_path,
            imports=imports,
            name=None,
            class_path=nested_path,
            line_start=class_start,
            line_end=header_end,
            kind="class",
            decorators=self._decorator_names(node),
            is_async=False,
        )
        return [class_chunk] + member_chunks

    def _build_python_chunk(
        self,
        lines: List[str],
        file_path: str,
        imports: List[str],
        name: Optional[str],
        class_path: Tuple[str, ...],
        line_start: int,
        line_end: int,
        kind: str,
        decorators: List[str],
        is_async: bool,
    ) -> CodeChunk:
        """
        Assemble a CodeChunk for a Python construct.

        Args:
            lines: The file's lines, used to slice exact source text.
            file_path: Path recorded on the chunk.
            imports: The file's import statements.
            name: Function or method name; None for a class chunk.
            class_path: Enclosing class names, outermost first.
            line_start: First line of the construct, 1-indexed.
            line_end: Last line of the construct, 1-indexed.
            kind: "function", "method", or "class".
            decorators: Decorator names applied to the construct.
            is_async: Whether the construct is an async function.

        Returns:
            The assembled CodeChunk.
        """
        class_name = ".".join(class_path) if class_path else None
        qualified = ".".join(filter(None, (class_name, name)))

        metadata: Dict[str, object] = {"chunk_strategy": "python_ast", "kind": kind}
        if decorators:
            metadata["decorators"] = decorators
        if is_async:
            metadata["is_async"] = True

        return CodeChunk(
            chunk_id=f"{file_path}::{qualified}",
            file_path=file_path,
            language="python",
            class_name=class_name,
            function_name=name,
            line_start=line_start,
            line_end=line_end,
            imports=imports,
            content=self._slice(lines, line_start, line_end),
            metadata=metadata,
        )

    def _collect_imports(self, tree: ast.Module, lines: List[str]) -> List[str]:
        """
        Extract the file's import statements as source text.

        Every import in the file is collected, including ones nested in
        functions or guarded by `try`/`if`, and the same list is
        attached to every chunk from that file. This is deliberately a
        file-level view rather than a per-scope resolution: a chunk
        holding one function does not contain the imports its code
        depends on, so without them the embedded text refers to names
        that appear nowhere in the chunk.

        Args:
            tree: The parsed module.
            lines: The file's lines, used to slice exact source text.

        Returns:
            Import statements in source order, deduplicated.
        """
        statements: List[Tuple[int, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                text = self._slice(lines, node.lineno, node.end_lineno or node.lineno)
                statements.append((node.lineno, text))

        seen = set()
        ordered: List[str] = []
        for _, text in sorted(statements, key=lambda item: item[0]):
            stripped = text.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                ordered.append(stripped)
        return ordered

    @staticmethod
    def _node_start(node: ast.AST) -> int:
        """
        Find the first line of a definition, including its decorators.

        `node.lineno` points at the `def`/`class` keyword, so decorators
        would otherwise fall outside the chunk. They belong inside it:
        a `@property` or `@staticmethod` changes what the code means,
        and a chunk that omits it misleads both the embedding and the
        reader.

        Args:
            node: A function or class definition node.

        Returns:
            The 1-indexed first line of the definition.
        """
        starts = [node.lineno]
        for decorator in getattr(node, "decorator_list", []):
            starts.append(decorator.lineno)
        return min(starts)

    @staticmethod
    def _decorator_names(node: ast.AST) -> List[str]:
        """
        Render a definition's decorators as readable names.

        Args:
            node: A function or class definition node.

        Returns:
            Decorator names, best-effort. Expressions that do not
            unparse cleanly are skipped rather than raising.
        """
        names: List[str] = []
        for decorator in getattr(node, "decorator_list", []):
            try:
                names.append(ast.unparse(decorator))
            except (AttributeError, ValueError):
                continue
        return names

    # ------------------------------------------------------------------
    # Markdown / plain text chunking
    # ------------------------------------------------------------------

    def _chunk_structured_text(
        self,
        content: str,
        file_path: str,
        language: str,
        strategy: Optional[str] = None,
        extra_metadata: Optional[Dict[str, object]] = None,
    ) -> List[CodeChunk]:
        """
        Chunk prose by its structure.

        Markdown splits at headings; anything else splits at paragraph
        boundaries. Sections longer than `max_chunk_lines` are packed
        into several chunks, split between paragraphs so a chunk never
        begins mid-sentence.

        Context is preserved through metadata, not text: a continuation
        chunk records the heading it belongs under and its position in
        the section, rather than having the heading pasted on top of it.
        Copying the heading into the text would break the guarantee that
        chunk content is an exact slice of the file.

        Args:
            content: The file's full text.
            file_path: Path recorded on each chunk.
            language: Language value recorded on each chunk.
            strategy: Override for the recorded `chunk_strategy`.
            extra_metadata: Extra metadata merged into every chunk,
                used to record why a fallback was taken.

        Returns:
            The content's chunks, in source order.
        """
        lines = content.split("\n")
        sections = (
            self._markdown_sections(lines)
            if language == "markdown"
            else [(None, 0, 1, len(lines))]
        )
        resolved_strategy = strategy or (
            "markdown_sections" if language == "markdown" else "text_paragraphs"
        )

        chunks: List[CodeChunk] = []
        for section_index, (heading, level, start, end) in enumerate(sections):
            ranges = self._pack_paragraphs(lines, start, end)
            for part_index, (chunk_start, chunk_end) in enumerate(ranges):
                text = self._slice(lines, chunk_start, chunk_end)
                if not text.strip():
                    continue

                metadata: Dict[str, object] = {
                    "chunk_strategy": resolved_strategy,
                    "section_index": section_index,
                }
                if heading is not None:
                    metadata["heading"] = heading
                    metadata["heading_level"] = level
                if len(ranges) > 1:
                    metadata["part"] = part_index + 1
                    metadata["part_count"] = len(ranges)
                if extra_metadata:
                    metadata.update(extra_metadata)

                chunks.append(
                    CodeChunk(
                        chunk_id=f"{file_path}::L{chunk_start}-L{chunk_end}",
                        file_path=file_path,
                        language=language,
                        class_name=None,
                        function_name=None,
                        line_start=chunk_start,
                        line_end=chunk_end,
                        imports=[],
                        content=text,
                        metadata=metadata,
                    )
                )
        return chunks

    def _markdown_sections(
        self, lines: List[str]
    ) -> List[Tuple[Optional[str], int, int, int]]:
        """
        Split Markdown into heading-delimited sections.

        Headings inside fenced code blocks are ignored. A README full of
        ```python fences would otherwise be shredded at every `# comment`
        line inside its examples.

        Args:
            lines: The file's lines.

        Returns:
            Tuples of (heading text, heading level, first line, last
            line), 1-indexed and inclusive. Content before the first
            heading becomes a leading section with a None heading.
        """
        boundaries: List[Tuple[int, str, int]] = []
        in_fence = False
        fence_marker = ""

        for index, line in enumerate(lines, start=1):
            fence = _FENCE_PATTERN.match(line)
            if fence:
                marker = fence.group(1)[0]
                if not in_fence:
                    in_fence, fence_marker = True, marker
                elif marker == fence_marker:
                    in_fence, fence_marker = False, ""
                continue

            if in_fence:
                continue

            heading = _HEADING_PATTERN.match(line)
            if heading:
                boundaries.append(
                    (index, heading.group(2).strip(), len(heading.group(1)))
                )

        if not boundaries:
            return [(None, 0, 1, len(lines))]

        sections: List[Tuple[Optional[str], int, int, int]] = []
        if boundaries[0][0] > 1:
            sections.append((None, 0, 1, boundaries[0][0] - 1))

        for position, (line_no, text, level) in enumerate(boundaries):
            end = (
                boundaries[position + 1][0] - 1
                if position + 1 < len(boundaries)
                else len(lines)
            )
            sections.append((text, level, line_no, end))
        return sections

    def _pack_paragraphs(
        self, lines: List[str], start: int, end: int
    ) -> List[Tuple[int, int]]:
        """
        Group a line range into chunk-sized ranges at paragraph breaks.

        Args:
            lines: The file's lines.
            start: First line of the range, 1-indexed.
            end: Last line of the range, 1-indexed.

        Returns:
            Chunk ranges as (first line, last line), 1-indexed and
            inclusive. Blank runs between paragraphs are dropped, so the
            ranges need not cover every line in the input range.
        """
        paragraphs = self._paragraph_ranges(lines, start, end)
        if not paragraphs:
            return []

        packed: List[Tuple[int, int]] = []
        current: Optional[Tuple[int, int]] = None
        current_length = 0

        for para_start, para_end in paragraphs:
            length = para_end - para_start + 1

            if length > self.max_chunk_lines:
                if current is not None:
                    packed.append(current)
                    current, current_length = None, 0
                packed.extend(self._hard_split(para_start, para_end))
                continue

            if current is None:
                current, current_length = (para_start, para_end), length
            elif current_length + length <= self.max_chunk_lines:
                current = (current[0], para_end)
                current_length += length
            else:
                packed.append(current)
                current, current_length = (para_start, para_end), length

        if current is not None:
            packed.append(current)
        return packed

    @staticmethod
    def _paragraph_ranges(
        lines: List[str], start: int, end: int
    ) -> List[Tuple[int, int]]:
        """
        Find runs of non-blank lines within a range.

        Args:
            lines: The file's lines.
            start: First line of the range, 1-indexed.
            end: Last line of the range, 1-indexed.

        Returns:
            Paragraph ranges as (first line, last line), 1-indexed and
            inclusive.
        """
        ranges: List[Tuple[int, int]] = []
        para_start: Optional[int] = None

        for index in range(start, min(end, len(lines)) + 1):
            if lines[index - 1].strip():
                if para_start is None:
                    para_start = index
            elif para_start is not None:
                ranges.append((para_start, index - 1))
                para_start = None

        if para_start is not None:
            ranges.append((para_start, min(end, len(lines))))
        return ranges

    def _hard_split(self, start: int, end: int) -> List[Tuple[int, int]]:
        """
        Split an oversized paragraph into fixed-size ranges.

        The last resort for content with no internal blank lines, such
        as a large table or a minified block.

        Args:
            start: First line of the paragraph, 1-indexed.
            end: Last line of the paragraph, 1-indexed.

        Returns:
            Consecutive ranges of at most `max_chunk_lines` lines.
        """
        return [
            (window, min(window + self.max_chunk_lines - 1, end))
            for window in range(start, end + 1, self.max_chunk_lines)
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_supported(self, file_path: str) -> str:
        """
        Validate a path and resolve its language.

        Args:
            file_path: Path to validate.

        Returns:
            The language name for the path.

        Raises:
            ValueError: If `file_path` is empty or whitespace only.
            UnsupportedFileTypeError: If the extension is out of scope.
        """
        if not file_path or not str(file_path).strip():
            raise ValueError("file_path must be a non-empty string.")

        language = self.language_for(file_path)
        if language is None:
            supported = ", ".join(sorted(self.SUPPORTED_LANGUAGES))
            raise UnsupportedFileTypeError(
                f"Cannot chunk {file_path!r}: only {supported} are supported."
            )
        return language

    def _require_filesystem(self) -> FilesystemTools:
        """
        Return the FilesystemTools instance, constructing it on first use.

        Returns:
            The FilesystemTools instance scoped to the configured
            workspace root.

        Raises:
            ToolExecutionError: If the workspace root is missing or is
                not a directory.
        """
        if self._filesystem is None:
            self._filesystem = FilesystemTools(config=self.config)
        return self._filesystem

    @staticmethod
    def _slice(lines: Sequence[str], start: int, end: int) -> str:
        """
        Extract an exact source slice by line range.

        Args:
            lines: The file's lines, split on newlines.
            start: First line to include, 1-indexed.
            end: Last line to include, 1-indexed and inclusive.

        Returns:
            The original text of those lines, joined with newlines and
            otherwise untouched. No trailing newline: the result is the
            file's exact substring from the start of `start` to the end
            of `end`.
        """
        return "\n".join(lines[start - 1 : end])

    @staticmethod
    def _ensure_unique_ids(chunks: List[CodeChunk]) -> List[CodeChunk]:
        """
        Disambiguate repeated chunk IDs in place.

        `chunk_id` is the vector store's primary key, so a collision
        silently overwrites a chunk instead of adding one. Names do
        repeat legitimately -- a method defined in both branches of an
        `if`, or two `@overload` stubs -- so the line number is appended
        to make the ID unique while keeping it stable across runs.

        Args:
            chunks: Chunks to check, in source order.

        Returns:
            The same list, with duplicate IDs rewritten.
        """
        seen: Dict[str, int] = {}
        for chunk in chunks:
            if chunk.chunk_id not in seen:
                seen[chunk.chunk_id] = 1
                continue

            seen[chunk.chunk_id] += 1
            candidate = f"{chunk.chunk_id}#L{chunk.line_start}"
            if candidate in seen:
                candidate = f"{candidate}-{seen[chunk.chunk_id]}"
            seen[candidate] = 1
            chunk.chunk_id = candidate
        return chunks
