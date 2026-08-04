"""
filesystem_tools.py
====================

Filesystem access for the codebase being analyzed: reading, writing,
listing, and searching files within a sandboxed workspace root.

Every path passed in is resolved against `workspace_root` and checked
before use, so a caller cannot read or write outside the workspace even
by passing an absolute path or a `..` traversal. Failures raise the
project's tool exceptions rather than bare OSErrors, so the notebook can
report them as readable messages.

Directory walks skip `Config.ignore_directories`, and reads are capped
at `Config.max_file_size_bytes`, so the Scope & Limits ceilings are
enforced here rather than restated by each caller.

TODO: Expose the read/list/search operations as MCP tools once the MCP
server is implemented.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

from ..config import Config
from ..exceptions.tool_exceptions import (
    EmptyFileError,
    FileTooLargeError,
    PathOutsideWorkspaceError,
    ToolExecutionError,
    UnsupportedFileTypeError,
)


class FilesystemTools:
    """
    Sandboxed filesystem operations scoped to a workspace root.

    Intended to be registered with the ToolRegistry so agents can read,
    list, and search the target codebase without reaching outside it.
    """

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> None:
        """
        Initialize FilesystemTools.

        Args:
            workspace_root: Root directory all operations are scoped to.
                Falls back to `config.workspace_root` when omitted.
            config: Optional Config instance. A default is loaded when
                not supplied.

        Raises:
            ToolExecutionError: If the workspace root does not exist or
                is not a directory.
        """
        self.config = config or Config.load()
        self.workspace_root = workspace_root or self.config.workspace_root

        root = Path(self.workspace_root).expanduser()
        try:
            self._root = root.resolve(strict=True)
        except (OSError, FileNotFoundError) as exc:
            raise ToolExecutionError(
                f"Workspace root does not exist: {self.workspace_root!r}"
            ) from exc

        if not self._root.is_dir():
            raise ToolExecutionError(
                f"Workspace root is not a directory: {self.workspace_root!r}"
            )

    # ------------------------------------------------------------------
    # Path validation
    # ------------------------------------------------------------------

    def resolve_path(self, path: str) -> Path:
        """
        Resolve a path against the workspace root and verify it is inside.

        Relative paths are interpreted relative to the workspace root;
        absolute paths are permitted only if they land inside it. This is
        the single gate every other method routes through.

        Args:
            path: Path to resolve, relative or absolute.

        Returns:
            The resolved absolute Path.

        Raises:
            ValueError: If `path` is empty or whitespace only.
            PathOutsideWorkspaceError: If the resolved path escapes the
                workspace root.
        """
        if not path or not str(path).strip():
            raise ValueError("path must be a non-empty string.")

        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self._root / candidate

        # resolve() collapses ".." segments, so traversal is caught below
        # rather than being smuggled through as a literal path component.
        resolved = candidate.resolve()

        if resolved != self._root and self._root not in resolved.parents:
            raise PathOutsideWorkspaceError(
                f"Path {path!r} resolves to {resolved}, which is outside "
                f"the workspace root {self._root}."
            )
        return resolved

    def is_within_workspace(self, path: str) -> bool:
        """
        Report whether a path lies inside the workspace root.

        Non-raising counterpart to `resolve_path`, for callers that want
        to test a path rather than act on it.

        Args:
            path: Path to test.

        Returns:
            True if the path resolves inside the workspace root.
        """
        try:
            self.resolve_path(path)
        except (ValueError, PathOutsideWorkspaceError):
            return False
        return True

    def relative_path(self, path: Path) -> str:
        """
        Express an absolute path relative to the workspace root.

        Args:
            path: Absolute path inside the workspace.

        Returns:
            A workspace-relative path using forward slashes, so results
            are identical on Windows and POSIX.
        """
        return path.relative_to(self._root).as_posix()

    # ------------------------------------------------------------------
    # Reading and writing
    # ------------------------------------------------------------------

    def file_exists(self, path: str) -> bool:
        """
        Check whether a path points at an existing file.

        Args:
            path: Path to check, relative to the workspace root.

        Returns:
            True if the path exists and is a regular file.

        Raises:
            PathOutsideWorkspaceError: If the path escapes the workspace.
        """
        return self.resolve_path(path).is_file()

    def read_file(self, path: str, allow_empty: bool = True) -> str:
        """
        Read a UTF-8 text file.

        Args:
            path: Path to the file, relative to the workspace root.
            allow_empty: When False, a blank file raises EmptyFileError.
                Defaults to True because empty files are legitimate in
                Python packages (an empty `__init__.py`); ingestion opts
                into the strict behavior when a file is expected to have
                content worth chunking.

        Returns:
            The file's contents.

        Raises:
            PathOutsideWorkspaceError: If the path escapes the workspace.
            ToolExecutionError: If the file is missing or unreadable.
            FileTooLargeError: If the file exceeds the configured size
                ceiling.
            UnsupportedFileTypeError: If the file is not valid UTF-8 text
                (in practice, a binary).
            EmptyFileError: If the file is blank and `allow_empty` is
                False.
        """
        target = self.resolve_path(path)

        if not target.is_file():
            raise ToolExecutionError(f"File not found: {path!r}")

        size = target.stat().st_size
        limit = self.config.max_file_size_bytes
        if size > limit:
            raise FileTooLargeError(
                f"File {path!r} is {size} bytes, exceeding the {limit} byte limit."
            )

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise UnsupportedFileTypeError(
                f"File {path!r} is not valid UTF-8 text and is likely binary."
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(f"Could not read {path!r}: {exc}") from exc

        if not allow_empty and not content.strip():
            raise EmptyFileError(f"File {path!r} is empty.")

        return content

    def write_file(self, path: str, content: str) -> bool:
        """
        Write UTF-8 text to a file, creating parent directories as needed.

        Args:
            path: Path to the file, relative to the workspace root.
            content: Content to write.

        Returns:
            True when the write succeeds.

        Raises:
            PathOutsideWorkspaceError: If the path escapes the workspace.
            ToolExecutionError: If the write fails.
        """
        target = self.resolve_path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ToolExecutionError(f"Could not write {path!r}: {exc}") from exc
        return True

    def delete_file(self, path: str) -> bool:
        """
        Delete a file.

        Args:
            path: Path to the file, relative to the workspace root.

        Returns:
            True if a file was deleted, False if nothing was there.

        Raises:
            PathOutsideWorkspaceError: If the path escapes the workspace.
            ToolExecutionError: If the path is a directory or deletion
                fails.
        """
        target = self.resolve_path(path)

        if not target.exists():
            return False
        if target.is_dir():
            raise ToolExecutionError(
                f"Refusing to delete {path!r}: it is a directory, not a file."
            )

        try:
            target.unlink()
        except OSError as exc:
            raise ToolExecutionError(f"Could not delete {path!r}: {exc}") from exc
        return True

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_files(
        self,
        directory: str = ".",
        pattern: Optional[str] = None,
        recursive: bool = False,
    ) -> List[str]:
        """
        List files in a directory.

        Args:
            directory: Directory to list, relative to the workspace root.
            pattern: Optional glob pattern (e.g. `*.py`) filtering by
                filename.
            recursive: When True, descend into subdirectories, skipping
                the configured ignore list.

        Returns:
            Sorted workspace-relative file paths.

        Raises:
            PathOutsideWorkspaceError: If the path escapes the workspace.
            ToolExecutionError: If the directory does not exist.
        """
        base = self._require_directory(directory)

        if recursive:
            candidates = list(self._walk_files(base))
        else:
            candidates = [entry for entry in base.iterdir() if entry.is_file()]

        if pattern:
            candidates = [entry for entry in candidates if entry.match(pattern)]

        return sorted(self.relative_path(entry) for entry in candidates)

    def list_directories(self, directory: str = ".") -> List[str]:
        """
        List the immediate subdirectories of a directory.

        Directories named in `Config.ignore_directories` are omitted.

        Args:
            directory: Directory to list, relative to the workspace root.

        Returns:
            Sorted workspace-relative directory paths.

        Raises:
            PathOutsideWorkspaceError: If the path escapes the workspace.
            ToolExecutionError: If the directory does not exist.
        """
        base = self._require_directory(directory)
        ignored = set(self.config.ignore_directories)
        return sorted(
            self.relative_path(entry)
            for entry in base.iterdir()
            if entry.is_dir() and entry.name not in ignored
        )

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def find_files_by_extension(
        self,
        extension: str,
        directory: str = ".",
    ) -> List[str]:
        """
        Find files with a given extension, recursively.

        Args:
            extension: Extension to match, with or without the leading
                dot (`py` and `.py` behave identically). Matched
                case-insensitively.
            directory: Directory to search under, relative to the
                workspace root.

        Returns:
            Sorted workspace-relative paths of matching files.

        Raises:
            ValueError: If `extension` is empty.
            PathOutsideWorkspaceError: If the path escapes the workspace.
            ToolExecutionError: If the directory does not exist.
        """
        if not extension or not extension.strip():
            raise ValueError("extension must be a non-empty string.")

        normalized = extension.strip().lower()
        if not normalized.startswith("."):
            normalized = f".{normalized}"

        base = self._require_directory(directory)
        return sorted(
            self.relative_path(entry)
            for entry in self._walk_files(base)
            if entry.suffix.lower() == normalized
        )

    def find_files_by_name(
        self,
        name: str,
        directory: str = ".",
        exact: bool = False,
    ) -> List[str]:
        """
        Find files by filename, recursively.

        Args:
            name: Filename or fragment to look for, matched
                case-insensitively.
            directory: Directory to search under, relative to the
                workspace root.
            exact: When True, the whole filename must equal `name`;
                otherwise a substring match is enough.

        Returns:
            Sorted workspace-relative paths of matching files.

        Raises:
            ValueError: If `name` is empty.
            PathOutsideWorkspaceError: If the path escapes the workspace.
            ToolExecutionError: If the directory does not exist.
        """
        if not name or not name.strip():
            raise ValueError("name must be a non-empty string.")

        needle = name.strip().lower()
        base = self._require_directory(directory)

        matches = []
        for entry in self._walk_files(base):
            filename = entry.name.lower()
            if (filename == needle) if exact else (needle in filename):
                matches.append(self.relative_path(entry))
        return sorted(matches)

    def search_codebase(
        self,
        query: str,
        directory: str = ".",
        extensions: Optional[Sequence[str]] = None,
        max_results: int = 100,
    ) -> List[str]:
        """
        Find files whose contents contain a query string.

        A plain case-insensitive substring search. Files that are too
        large, binary, or unreadable are skipped rather than raising, so
        one bad file cannot abort a search across the repository.

        Args:
            query: Text to search for (e.g. a function name).
            directory: Directory to search under, relative to the
                workspace root.
            extensions: Optional extensions to restrict the search to.
            max_results: Cap on the number of matching files returned.

        Returns:
            Sorted workspace-relative paths of files containing `query`.

        Raises:
            ValueError: If `query` is empty or `max_results` is not
                positive.
            PathOutsideWorkspaceError: If the path escapes the workspace.
            ToolExecutionError: If the directory does not exist.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if max_results <= 0:
            raise ValueError("max_results must be positive.")

        needle = query.lower()
        allowed = self._normalize_extensions(extensions)
        base = self._require_directory(directory)

        matches: List[str] = []
        for entry in self._walk_files(base):
            if allowed is not None and entry.suffix.lower() not in allowed:
                continue

            relative = self.relative_path(entry)
            try:
                content = self.read_file(relative)
            except (FileTooLargeError, UnsupportedFileTypeError, ToolExecutionError):
                # Skipping is the point: an unreadable file is not a
                # search failure.
                continue

            if needle in content.lower():
                matches.append(relative)
                if len(matches) >= max_results:
                    break

        return sorted(matches)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_directory(self, directory: str) -> Path:
        """
        Resolve a path and confirm it is an existing directory.

        Args:
            directory: Path to resolve.

        Returns:
            The resolved directory Path.

        Raises:
            ToolExecutionError: If the path is missing or not a
                directory.
        """
        base = self.resolve_path(directory)
        if not base.is_dir():
            raise ToolExecutionError(f"Directory not found: {directory!r}")
        return base

    def _walk_files(self, base: Path) -> Iterator[Path]:
        """
        Yield every file under a directory, skipping ignored directories.

        Args:
            base: Directory to walk.

        Yields:
            Absolute paths of files found beneath `base`.
        """
        ignored = set(self.config.ignore_directories)
        for dirpath, dirnames, filenames in os.walk(base):
            # Pruning in place stops os.walk descending into these at all.
            dirnames[:] = [name for name in dirnames if name not in ignored]
            for filename in filenames:
                yield Path(dirpath) / filename

    @staticmethod
    def _normalize_extensions(
        extensions: Optional[Sequence[str]],
    ) -> Optional[set]:
        """
        Normalize an extension filter to lowercase, dot-prefixed form.

        Args:
            extensions: Extensions to normalize, or None.

        Returns:
            A set of normalized extensions, or None when unfiltered.
        """
        if extensions is None:
            return None
        normalized = set()
        for extension in extensions:
            value = extension.strip().lower()
            if not value:
                continue
            normalized.add(value if value.startswith(".") else f".{value}")
        return normalized
