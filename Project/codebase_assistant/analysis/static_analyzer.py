"""
static_analyzer.py
===================

Deterministic first pass of the bug detection pipeline.

Runs `pyflakes` and Python's `ast` over source files before any LLM is
consulted. Because it is plain static analysis it cannot hallucinate, so
every finding is labelled `detection_method="static"` and carries the
highest confidence in the system. Nothing here calls a model, touches
the retriever, or reaches the network.

Every report's `evidence` is an exact slice of the source at the
recorded line range, not a paraphrase. That is deliberate: the whole
point of the grounding stage is that a claim can be checked against the
file, and a report whose evidence was reformatted cannot be.

The checks lean conservative. A false positive from a deterministic pass
is worse than a miss, because these findings are weighted above anything
the LLM proposes -- so a check that cannot be sure stays quiet. That is
why `@property` setters are excluded from duplicate detection and why
argument-count checking gives up the moment a name might be rebound.

TODO: Move `MAX_EVIDENCE_LINES` and the severity table onto Config once
the categories have settled.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..config import Config
from ..exceptions.analysis_exceptions import StaticAnalysisError
from ..exceptions.base import CodebaseAssistantError
from ..schemas.schemas import BugReport, CodeChunk
from ..tools.filesystem_tools import FilesystemTools

logger = logging.getLogger(__name__)

#: Recorded as `function_name` when a finding sits outside any function.
#: BugReport requires the field, and an empty string reads as missing
#: data rather than as a deliberate "module level".
MODULE_SCOPE = "<module>"

#: Ceiling on how many lines of evidence a single report quotes, so one
#: finding on a large block cannot carry an entire file into a prompt.
MAX_EVIDENCE_LINES = 12

#: Markers `find_todo_markers` looks for.
_TODO_PATTERN = re.compile(r"#\s*(TODO|FIXME|XXX|HACK)\b[:\s]*(.*)", re.IGNORECASE)

#: Statements after which the rest of a block cannot execute.
_TERMINATORS = (ast.Return, ast.Raise, ast.Break, ast.Continue)

#: How each pyflakes message maps onto a BugReport.
#: (bug_type, severity, confidence). Confidence expresses how likely the
#: finding is to be a real defect, not how sure the checker is that it
#: matched -- pyflakes is certain either way.
_PYFLAKES_MAP: Dict[str, Tuple[str, str, float]] = {
    "UndefinedName": ("undefined_variable", "high", 0.95),
    "UndefinedLocal": ("undefined_variable", "high", 0.95),
    "UndefinedExport": ("undefined_variable", "medium", 0.85),
    "UnusedImport": ("unused_import", "low", 0.95),
    "UnusedVariable": ("unused_variable", "low", 0.85),
    "UnusedIndirectAssignment": ("unused_variable", "low", 0.80),
    "ImportStarUsed": ("wildcard_import", "low", 0.80),
    "ImportStarUsage": ("wildcard_import", "medium", 0.80),
    "ImportShadowedByLoopVar": ("shadowed_import", "medium", 0.85),
    "RedefinedWhileUnused": ("duplicate_definition", "medium", 0.85),
    "MultiValueRepeatedKeyLiteral": ("duplicate_dict_key", "medium", 0.90),
    "MultiValueRepeatedKeyVariable": ("duplicate_dict_key", "medium", 0.85),
    "FStringMissingPlaceholders": ("suspicious_fstring", "low", 0.80),
    "IsLiteral": ("identity_comparison", "medium", 0.90),
    "AssertTuple": ("always_true_assert", "high", 0.95),
    "RaiseNotImplemented": ("wrong_exception", "medium", 0.90),
    "TwoStarredExpressions": ("syntax_misuse", "high", 0.95),
    "ReturnOutsideFunction": ("syntax_misuse", "high", 0.95),
    "YieldOutsideFunction": ("syntax_misuse", "high", 0.95),
    "ContinueOutsideLoop": ("syntax_misuse", "high", 0.95),
    "BreakOutsideLoop": ("syntax_misuse", "high", 0.95),
    "DefaultExceptNotLast": ("unreachable_except", "high", 0.95),
}

#: Applied to any pyflakes message not named above, so a check gained in
#: a future pyflakes release surfaces instead of being dropped.
_PYFLAKES_FALLBACK: Tuple[str, str, float] = ("code_quality", "low", 0.75)


@dataclass
class AnalysisReport:
    """
    Outcome of a repository-wide static analysis run.

    Attributes:
        findings: Every BugReport produced, ordered by file and line.
        files_analyzed: Files successfully parsed and checked.
        skipped: Files deliberately not analyzed, as (path, reason).
            Oversized and unreadable files land here.
        failed: Files where analysis itself broke, as (path, reason).
            Distinct from a *detected* syntax error, which is a finding.
        limit_reached: The Config ceiling that stopped the run early, or
            None if it ran to completion.
    """

    findings: List[BugReport] = field(default_factory=list)
    files_analyzed: int = 0
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)
    limit_reached: Optional[str] = None

    def summary(self) -> str:
        """
        Render a one-line summary of the run.

        Returns:
            A readable summary suitable for logs or a notebook cell.
        """
        parts = [
            f"{len(self.findings)} finding(s)",
            f"{self.files_analyzed} file(s) analyzed",
        ]
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if self.limit_reached:
            parts.append(f"stopped at limit: {self.limit_reached}")
        return ", ".join(parts)

    def by_type(self) -> Dict[str, int]:
        """
        Count findings by bug type.

        Returns:
            A mapping of bug type to occurrence count, most common
            first.
        """
        counts: Dict[str, int] = {}
        for finding in self.findings:
            counts[finding.bug_type] = counts.get(finding.bug_type, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))


class StaticAnalyzer:
    """
    Deterministic static analysis pass over Python source files.

    Produces the `static` half of the bug pipeline. Findings from here
    are intended to be weighted above anything an LLM proposes.
    """

    def __init__(
        self,
        workspace_root: str = ".",
        config: Optional[Config] = None,
        filesystem: Optional[FilesystemTools] = None,
    ) -> None:
        """
        Initialize the StaticAnalyzer.

        Args:
            workspace_root: Root directory all analyzed paths are
                resolved against. Kept as the leading parameter to match
                the scaffold's original signature.
            config: Optional Config instance. A default is loaded when
                not supplied.
            filesystem: Optional FilesystemTools. Built from the config
                and workspace root when omitted, so the sandbox check,
                the size ceiling, and the ignore list are enforced in
                one place rather than restated here.

        Raises:
            ToolExecutionError: If the workspace root does not exist.
        """
        self.workspace_root = workspace_root
        self.config = config or Config.load()
        self.filesystem = filesystem or FilesystemTools(
            workspace_root=workspace_root, config=self.config
        )

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def analyze_file(
        self, file_path: str, include_todos: bool = False
    ) -> List[BugReport]:
        """
        Run every static check against a single file.

        The file is read and parsed once here, then handed to each
        check. The individual `check_*` methods below re-read for
        callers that want one check in isolation; this path does not pay
        that cost repeatedly.

        Args:
            file_path: Python file to analyze, relative to the workspace
                root.
            include_todos: When True, TODO and FIXME markers are
                reported as findings. Off by default because a healthy
                codebase is full of them and they would drown the
                genuine defects.

        Returns:
            Findings ordered by line number. A file that will not parse
            yields exactly one syntax-error finding, since no other
            check can run on it.

        Raises:
            ValueError: If `file_path` is empty.
            PathOutsideWorkspaceError: If the path escapes the
                workspace.
            FileTooLargeError: If the file exceeds the size ceiling.
            ToolExecutionError: If the file is missing or unreadable.
        """
        return self._analyze_source(
            file_path, self._read(file_path), include_todos=include_todos
        )

    def analyze_repository(
        self, repo_path: str = ".", include_todos: bool = False
    ) -> List[BugReport]:
        """
        Run every static check across an entire repository.

        Args:
            repo_path: Directory to analyze, relative to the workspace
                root.
            include_todos: When True, TODO and FIXME markers are
                reported.

        Returns:
            Every finding, ordered by file and line.

        Raises:
            PathOutsideWorkspaceError: If the path escapes the
                workspace.
            ToolExecutionError: If the directory does not exist.
        """
        return self.analyze_repository_detailed(
            repo_path, include_todos=include_todos
        ).findings

    def analyze_repository_detailed(
        self, repo_path: str = ".", include_todos: bool = False
    ) -> AnalysisReport:
        """
        Analyze a repository and report what was skipped or failed.

        The reporting counterpart to `analyze_repository`, for callers
        that need to know a file was passed over rather than silently
        finding nothing in it. A clean result over a repository where
        half the files were unreadable is not a clean repository.

        Files are discovered through FilesystemTools, so
        `Config.ignore_directories` is honored without being restated,
        and the `Config` ceilings on file count and total lines apply
        across the run.

        Args:
            repo_path: Directory to analyze, relative to the workspace
                root.
            include_todos: When True, TODO and FIXME markers are
                reported.

        Returns:
            The findings plus per-file skip and failure reasons.

        Raises:
            PathOutsideWorkspaceError: If the path escapes the
                workspace.
            ToolExecutionError: If the directory does not exist.
        """
        report = AnalysisReport()
        candidates = self.filesystem.find_files_by_extension("py", repo_path)
        lines_seen = 0

        for file_path in candidates:
            limit = self._limit_reached(report.files_analyzed, lines_seen)
            if limit is not None:
                report.limit_reached = limit
                for pending in candidates[candidates.index(file_path):]:
                    report.skipped.append((pending, f"limit: {limit}"))
                break

            try:
                source = self._read(file_path)
            except CodebaseAssistantError as exc:
                # Oversized, binary, or unreadable. FilesystemTools has
                # already classified it, so its message is the reason.
                report.skipped.append((file_path, str(exc)))
                logger.info("Skipped %s: %s", file_path, exc)
                continue

            try:
                findings = self._analyze_source(
                    file_path, source, include_todos=include_todos
                )
            except Exception as exc:
                # One malformed file must not end the run.
                report.failed.append((file_path, f"{type(exc).__name__}: {exc}"))
                logger.warning("Analysis failed on %s: %s", file_path, exc)
                continue

            report.findings.extend(findings)
            report.files_analyzed += 1
            lines_seen += source.count("\n") + 1

        report.findings = self._finalize(report.findings)
        return report

    def analyze_chunks(
        self, chunks: Sequence[CodeChunk], include_todos: bool = False
    ) -> List[BugReport]:
        """
        Analyze the files a set of retrieved chunks came from.

        The bridge from retrieval to analysis: hand it what the
        Retriever returned and it checks those files.

        Analysis is per *file*, not per chunk, and that is a constraint
        rather than a choice. A chunk holding one method is not a
        parseable module -- its body is indented and its imports live
        elsewhere -- so `ast` and pyflakes cannot run on it directly.

        Args:
            chunks: Chunks whose files should be analyzed. Duplicate
                files are analyzed once.
            include_todos: When True, TODO and FIXME markers are
                reported.

        Returns:
            Findings across those files, ordered by file and line.
        """
        seen: Set[str] = set()
        findings: List[BugReport] = []

        for chunk in chunks:
            if chunk.language != "python" or chunk.file_path in seen:
                continue
            seen.add(chunk.file_path)

            try:
                findings.extend(
                    self.analyze_file(chunk.file_path, include_todos=include_todos)
                )
            except CodebaseAssistantError as exc:
                logger.info("Skipped %s: %s", chunk.file_path, exc)

        return self._finalize(findings)

    # ------------------------------------------------------------------
    # Individual checks (public, per the scaffold's interface)
    # ------------------------------------------------------------------

    def check_syntax(
        self, file_path: str, source: Optional[str] = None
    ) -> List[BugReport]:
        """
        Detect syntax errors.

        Runs first, because every other check needs an AST.

        A syntax error is a *finding*, not a failure: the file is
        genuinely broken and the user wants to hear about it. That is
        the distinction `AnalysisReport.failed` draws.

        Args:
            file_path: File to check, relative to the workspace root.
            source: Pre-read source, to avoid a second read.

        Returns:
            One finding if the file will not parse, otherwise none.

        Raises:
            ValueError: If `file_path` is empty.
            ToolExecutionError: If the file cannot be read.
        """
        text = source if source is not None else self._read(file_path)
        lines = text.split("\n")

        try:
            ast.parse(text)
        except SyntaxError as exc:
            line = self._clamp(exc.lineno or 1, len(lines))
            detail = exc.msg or "invalid syntax"
            return [
                self._report(
                    bug_type="syntax_error",
                    description=(
                        f"{file_path} cannot be parsed: {detail} at line {line}."
                    ),
                    severity="high",
                    confidence=1.0,
                    file_path=file_path,
                    function_name=MODULE_SCOPE,
                    line_start=line,
                    line_end=line,
                    lines=lines,
                    suggested_fix="Fix the syntax error so the file can be parsed.",
                )
            ]
        except ValueError as exc:
            # Source containing null bytes, for example.
            return [
                self._report(
                    bug_type="syntax_error",
                    description=f"{file_path} cannot be parsed: {exc}.",
                    severity="high",
                    confidence=1.0,
                    file_path=file_path,
                    function_name=MODULE_SCOPE,
                    line_start=1,
                    line_end=1,
                    lines=lines,
                )
            ]

        return []

    def check_imports(self, file_path: str) -> List[BugReport]:
        """
        Detect unused, shadowed, and wildcard imports.

        Args:
            file_path: File to check, relative to the workspace root.

        Returns:
            Import-related findings.

        Raises:
            ValueError: If `file_path` is empty.
            ToolExecutionError: If the file cannot be read.
        """
        return self._filtered(
            file_path,
            {"unused_import", "wildcard_import", "shadowed_import"},
        )

    def check_undefined_names(self, file_path: str) -> List[BugReport]:
        """
        Detect undefined variables.

        Args:
            file_path: File to check, relative to the workspace root.

        Returns:
            Findings for names used before they exist.

        Raises:
            ValueError: If `file_path` is empty.
            ToolExecutionError: If the file cannot be read.
        """
        return self._filtered(file_path, {"undefined_variable"})

    def check_argument_counts(self, file_path: str) -> List[BugReport]:
        """
        Detect calls made with the wrong number of arguments.

        Args:
            file_path: File to check, relative to the workspace root.

        Returns:
            Findings for call sites that cannot match their definition.

        Raises:
            ValueError: If `file_path` is empty.
            ToolExecutionError: If the file cannot be read.
        """
        return self._run_one(file_path, self._argument_counts)

    def check_unreachable_code(self, file_path: str) -> List[BugReport]:
        """
        Detect statements that can never execute.

        Args:
            file_path: File to check, relative to the workspace root.

        Returns:
            Findings for dead code.

        Raises:
            ValueError: If `file_path` is empty.
            ToolExecutionError: If the file cannot be read.
        """
        return self._run_one(file_path, self._unreachable)

    def check_duplicate_definitions(self, file_path: str) -> List[BugReport]:
        """
        Detect functions or classes defined twice in the same scope.

        Args:
            file_path: File to check, relative to the workspace root.

        Returns:
            Findings for redefinitions that silently discard the first
            definition.

        Raises:
            ValueError: If `file_path` is empty.
            ToolExecutionError: If the file cannot be read.
        """
        return self._run_one(file_path, self._duplicate_definitions)

    def check_code_quality(self, file_path: str) -> List[BugReport]:
        """
        Detect the deterministic code-quality issues.

        Covers mutable default arguments and bare `except:` clauses --
        both mechanically detectable, both real defects rather than
        style preferences.

        Args:
            file_path: File to check, relative to the workspace root.

        Returns:
            Code-quality findings.

        Raises:
            ValueError: If `file_path` is empty.
            ToolExecutionError: If the file cannot be read.
        """
        return self._finalize(
            self._run_one(file_path, self._mutable_defaults)
            + self._run_one(file_path, self._bare_excepts)
        )

    def find_todo_markers(self, file_path: str) -> List[BugReport]:
        """
        Detect TODO, FIXME, XXX, and HACK markers.

        Args:
            file_path: File to check, relative to the workspace root.

        Returns:
            One finding per marker.

        Raises:
            ValueError: If `file_path` is empty.
            ToolExecutionError: If the file cannot be read.
        """
        source = self._read(file_path)
        lines = source.split("\n")

        try:
            scopes = self._function_index(ast.parse(source))
        except (SyntaxError, ValueError):
            # Markers are found by text, so a broken file is still fine;
            # only the enclosing-function attribution is lost.
            scopes = []

        return self._todo_markers(file_path, lines, scopes)

    # ------------------------------------------------------------------
    # pyflakes
    # ------------------------------------------------------------------

    def _pyflakes_findings(
        self,
        file_path: str,
        tree: ast.Module,
        lines: List[str],
        scopes: List[Tuple[int, int, str]],
    ) -> List[BugReport]:
        """
        Run pyflakes and convert its messages into BugReports.

        pyflakes is used through its Checker rather than its command
        line so the already-parsed AST can be reused and the results
        arrive as typed message objects instead of text needing a
        regex.

        Args:
            file_path: File being analyzed.
            tree: Its parsed AST.
            lines: Its source lines.
            scopes: Function ranges, for attributing findings.

        Returns:
            One finding per pyflakes message.

        Raises:
            StaticAnalysisError: If pyflakes is not installed.
        """
        checker_cls = self._import_pyflakes()

        try:
            checker = checker_cls(tree, filename=file_path)
        except Exception as exc:
            raise StaticAnalysisError(
                f"pyflakes could not analyze {file_path!r}: {exc}"
            ) from exc

        findings: List[BugReport] = []
        for message in checker.messages:
            name = type(message).__name__
            bug_type, severity, confidence = _PYFLAKES_MAP.get(
                name, _PYFLAKES_FALLBACK
            )
            line = self._clamp(getattr(message, "lineno", 1), len(lines))

            findings.append(
                self._report(
                    bug_type=bug_type,
                    description=self._describe(message),
                    severity=severity,
                    confidence=confidence,
                    file_path=file_path,
                    function_name=self._enclosing(scopes, line),
                    line_start=line,
                    line_end=line,
                    lines=lines,
                )
            )
        return findings

    @staticmethod
    def _import_pyflakes() -> Any:
        """
        Import the pyflakes Checker at call time.

        Deferred rather than imported at module scope so the package
        stays importable, and the rest of the scaffold runnable, without
        pyflakes installed.

        Returns:
            The pyflakes Checker class.

        Raises:
            StaticAnalysisError: If pyflakes is not installed.
        """
        try:
            from pyflakes.checker import Checker
        except ImportError as exc:
            raise StaticAnalysisError(
                "pyflakes is not installed. Install it with "
                "`pip install pyflakes` to run static analysis."
            ) from exc
        return Checker

    @staticmethod
    def _describe(message: Any) -> str:
        """
        Render a pyflakes message as a sentence.

        Args:
            message: A pyflakes message object.

        Returns:
            The formatted message text, without the file:line prefix
            pyflakes normally prepends -- BugReport already carries
            both.
        """
        try:
            return str(message.message % message.message_args)
        except (AttributeError, TypeError):
            return str(message)

    # ------------------------------------------------------------------
    # AST checks
    # ------------------------------------------------------------------

    def _unreachable(
        self,
        file_path: str,
        tree: ast.Module,
        lines: List[str],
        scopes: List[Tuple[int, int, str]],
    ) -> List[BugReport]:
        """
        Find statements following a return, raise, break, or continue.

        Only flags code in the *same* block as the terminator, which is
        what makes the check safe: anything at a different indentation
        level may still be reachable through another path.

        Args:
            file_path: File being analyzed.
            tree: Its parsed AST.
            lines: Its source lines.
            scopes: Function ranges, for attributing findings.

        Returns:
            One finding per unreachable run of statements.
        """
        findings: List[BugReport] = []

        for node in ast.walk(tree):
            for body in self._statement_blocks(node):
                for position, statement in enumerate(body[:-1]):
                    if not isinstance(statement, _TERMINATORS):
                        continue

                    dead = body[position + 1 :]
                    start = dead[0].lineno
                    end = max(
                        getattr(item, "end_lineno", item.lineno) or item.lineno
                        for item in dead
                    )
                    keyword = type(statement).__name__.lower()

                    findings.append(
                        self._report(
                            bug_type="unreachable_code",
                            description=(
                                f"Code after `{keyword}` on line "
                                f"{statement.lineno} can never execute."
                            ),
                            severity="medium",
                            confidence=0.90,
                            file_path=file_path,
                            function_name=self._enclosing(scopes, start),
                            line_start=start,
                            line_end=self._clamp(end, len(lines)),
                            lines=lines,
                            suggested_fix=(
                                "Remove the dead code, or move it above the "
                                f"`{keyword}`."
                            ),
                        )
                    )
                    break  # one report per block is enough

        return findings

    def _duplicate_definitions(
        self,
        file_path: str,
        tree: ast.Module,
        lines: List[str],
        scopes: List[Tuple[int, int, str]],
    ) -> List[BugReport]:
        """
        Find functions or classes defined twice in one scope.

        The second definition silently replaces the first, so the code
        that was written first never runs.

        Property setters, `typing.overload` stubs, and `singledispatch`
        registrations legitimately reuse a name, so any redefinition
        decorated with `@<name>.something` or `@overload` is ignored.
        Without that exclusion the check would fire on almost every
        well-written class.

        Args:
            file_path: File being analyzed.
            tree: Its parsed AST.
            lines: Its source lines.
            scopes: Function ranges, for attributing findings.

        Returns:
            One finding per redefinition.
        """
        findings: List[BugReport] = []
        definable = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

        for node in ast.walk(tree):
            for body in self._statement_blocks(node):
                first_seen: Dict[str, int] = {}

                for statement in body:
                    if not isinstance(statement, definable):
                        continue

                    name = statement.name
                    if name not in first_seen:
                        first_seen[name] = statement.lineno
                        continue

                    if self._is_intentional_redefinition(statement, name):
                        continue

                    kind = (
                        "Class" if isinstance(statement, ast.ClassDef) else "Function"
                    )
                    findings.append(
                        self._report(
                            bug_type="duplicate_definition",
                            description=(
                                f"{kind} `{name}` is redefined here; the "
                                f"definition on line {first_seen[name]} is "
                                f"discarded and never runs."
                            ),
                            severity="medium",
                            confidence=0.90,
                            file_path=file_path,
                            function_name=self._enclosing(scopes, statement.lineno),
                            line_start=statement.lineno,
                            line_end=self._clamp(
                                statement.end_lineno or statement.lineno, len(lines)
                            ),
                            lines=lines,
                            suggested_fix=(
                                f"Rename one of the two `{name}` definitions, or "
                                f"delete the one that is not needed."
                            ),
                        )
                    )

        return findings

    @staticmethod
    def _is_intentional_redefinition(node: ast.AST, name: str) -> bool:
        """
        Decide whether a redefinition is a deliberate language idiom.

        Args:
            node: The redefining node.
            name: The name being redefined.

        Returns:
            True for property setters, overloads, and dispatch
            registrations.
        """
        for decorator in getattr(node, "decorator_list", []):
            try:
                text = ast.unparse(decorator)
            except (AttributeError, ValueError):
                return True  # cannot tell, so stay quiet

            if text.startswith(f"{name}.") or text.endswith("overload"):
                return True
            if text.endswith(".register") or text.endswith(".setter"):
                return True
        return False

    def _argument_counts(
        self,
        file_path: str,
        tree: ast.Module,
        lines: List[str],
        scopes: List[Tuple[int, int, str]],
    ) -> List[BugReport]:
        """
        Find calls that cannot match their function's signature.

        Deliberately narrow. Only module-level functions defined exactly
        once, never reassigned, and never decorated are considered, and
        only calls made through a bare name in the same file. A
        decorator can change a signature, a reassignment can change what
        the name refers to, and a cross-module call cannot be resolved
        without imports -- so all of those are left alone rather than
        guessed at.

        Args:
            file_path: File being analyzed.
            tree: Its parsed AST.
            lines: Its source lines.
            scopes: Function ranges, for attributing findings.

        Returns:
            One finding per mismatched call site.
        """
        definitions = self._resolvable_functions(tree)
        if not definitions:
            return []

        findings: List[BugReport] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue

            definition = definitions.get(node.func.id)
            if definition is None:
                continue

            problem = self._call_mismatch(node, definition)
            if problem is None:
                continue

            line = node.lineno
            findings.append(
                self._report(
                    bug_type="missing_argument",
                    description=(
                        f"Call to `{node.func.id}` {problem}. It is defined on "
                        f"line {definition.lineno}."
                    ),
                    severity="high",
                    confidence=0.90,
                    file_path=file_path,
                    function_name=self._enclosing(scopes, line),
                    line_start=line,
                    line_end=self._clamp(
                        node.end_lineno or line, len(lines)
                    ),
                    lines=lines,
                    suggested_fix=(
                        f"Match the call to the signature of `{node.func.id}`."
                    ),
                )
            )
        return findings

    @staticmethod
    def _resolvable_functions(tree: ast.Module) -> Dict[str, ast.FunctionDef]:
        """
        Collect module-level functions whose signature can be trusted.

        Args:
            tree: The parsed module.

        Returns:
            A mapping of name to definition, excluding anything
            decorated, defined more than once, or rebound by an
            assignment anywhere in the file.
        """
        candidates: Dict[str, ast.FunctionDef] = {}
        rejected: Set[str] = set()

        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if statement.name in candidates or statement.decorator_list:
                    rejected.add(statement.name)
                elif isinstance(statement, ast.FunctionDef):
                    candidates[statement.name] = statement
                else:
                    rejected.add(statement.name)

        # Any assignment to the name means the call site may not be
        # reaching the definition found above.
        for node in ast.walk(tree):
            targets: List[ast.AST] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    rejected.add(target.id)

        return {
            name: node for name, node in candidates.items() if name not in rejected
        }

    @staticmethod
    def _call_mismatch(
        call: ast.Call, definition: ast.FunctionDef
    ) -> Optional[str]:
        """
        Compare one call site against a resolved signature.

        Args:
            call: The call node.
            definition: The function it resolves to.

        Returns:
            A phrase describing the mismatch, or None when the call is
            valid or cannot be judged with certainty.
        """
        if any(isinstance(arg, ast.Starred) for arg in call.args):
            return None
        if any(keyword.arg is None for keyword in call.keywords):
            return None

        spec = definition.args
        positional = list(spec.posonlyargs) + list(spec.args)
        names = [arg.arg for arg in positional]
        required = len(positional) - len(spec.defaults)

        supplied = len(call.args)
        by_keyword = {keyword.arg for keyword in call.keywords}

        known = set(names) | {arg.arg for arg in spec.kwonlyargs}
        if not spec.kwarg and not by_keyword.issubset(known):
            unexpected = sorted(by_keyword - known)
            return f"passes unexpected keyword argument(s) {', '.join(unexpected)}"

        if supplied > len(positional) and not spec.vararg:
            return (
                f"passes {supplied} positional argument(s) but at most "
                f"{len(positional)} are accepted"
            )

        filled = set(names[:supplied]) | (by_keyword & set(names))
        missing = [name for name in names[:required] if name not in filled]
        if missing:
            return f"is missing required argument(s) {', '.join(missing)}"

        required_kwonly = [
            arg.arg
            for arg, default in zip(spec.kwonlyargs, spec.kw_defaults)
            if default is None
        ]
        missing_kwonly = [
            name for name in required_kwonly if name not in by_keyword
        ]
        if missing_kwonly and not spec.kwarg:
            return (
                f"is missing required keyword argument(s) "
                f"{', '.join(missing_kwonly)}"
            )

        return None

    def _mutable_defaults(
        self,
        file_path: str,
        tree: ast.Module,
        lines: List[str],
        scopes: List[Tuple[int, int, str]],
    ) -> List[BugReport]:
        """
        Find mutable default arguments.

        A default is evaluated once at definition time, so a list or
        dict default is shared by every call and accumulates state
        between them. Classic, silent, and mechanically detectable.

        Args:
            file_path: File being analyzed.
            tree: Its parsed AST.
            lines: Its source lines.
            scopes: Function ranges, for attributing findings.

        Returns:
            One finding per mutable default.
        """
        findings: List[BugReport] = []
        mutable_calls = {"list", "dict", "set", "bytearray", "collections.OrderedDict"}

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            defaults = list(node.args.defaults) + [
                default for default in node.args.kw_defaults if default is not None
            ]
            for default in defaults:
                literal = isinstance(default, (ast.List, ast.Dict, ast.Set))
                constructor = (
                    isinstance(default, ast.Call)
                    and isinstance(default.func, ast.Name)
                    and default.func.id in mutable_calls
                    and not default.args
                )
                if not (literal or constructor):
                    continue

                line = self._clamp(default.lineno, len(lines))
                findings.append(
                    self._report(
                        bug_type="mutable_default_argument",
                        description=(
                            f"`{node.name}` has a mutable default argument. It "
                            f"is created once at definition time and shared by "
                            f"every call, so changes persist between calls."
                        ),
                        severity="medium",
                        confidence=0.85,
                        file_path=file_path,
                        function_name=self._enclosing(scopes, line),
                        line_start=line,
                        line_end=line,
                        lines=lines,
                        suggested_fix=(
                            "Default to None and create the value inside the "
                            "function body."
                        ),
                    )
                )
        return findings

    def _bare_excepts(
        self,
        file_path: str,
        tree: ast.Module,
        lines: List[str],
        scopes: List[Tuple[int, int, str]],
    ) -> List[BugReport]:
        """
        Find bare `except:` clauses.

        A bare except swallows KeyboardInterrupt and SystemExit along
        with the error it meant to handle, which turns a hang into an
        unkillable one.

        Args:
            file_path: File being analyzed.
            tree: Its parsed AST.
            lines: Its source lines.
            scopes: Function ranges, for attributing findings.

        Returns:
            One finding per bare except.
        """
        findings: List[BugReport] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or node.type is not None:
                continue

            line = self._clamp(node.lineno, len(lines))
            findings.append(
                self._report(
                    bug_type="bare_except",
                    description=(
                        "Bare `except:` catches every exception, including "
                        "KeyboardInterrupt and SystemExit."
                    ),
                    severity="low",
                    confidence=0.80,
                    file_path=file_path,
                    function_name=self._enclosing(scopes, line),
                    line_start=line,
                    line_end=line,
                    lines=lines,
                    suggested_fix="Catch a specific exception type, or `Exception`.",
                )
            )
        return findings

    def _todo_markers(
        self,
        file_path: str,
        lines: List[str],
        scopes: List[Tuple[int, int, str]],
    ) -> List[BugReport]:
        """
        Find TODO, FIXME, XXX, and HACK markers.

        Args:
            file_path: File being analyzed.
            lines: Its source lines.
            scopes: Function ranges, for attributing findings.

        Returns:
            One finding per marker.
        """
        findings: List[BugReport] = []

        for number, text in enumerate(lines, start=1):
            match = _TODO_PATTERN.search(text)
            if not match:
                continue

            marker = match.group(1).upper()
            note = match.group(2).strip() or "(no detail given)"
            findings.append(
                self._report(
                    bug_type="todo_marker",
                    description=f"{marker} left in the source: {note}",
                    severity="low",
                    confidence=1.0,
                    file_path=file_path,
                    function_name=self._enclosing(scopes, number),
                    line_start=number,
                    line_end=number,
                    lines=lines,
                )
            )
        return findings

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _read(self, file_path: str) -> str:
        """
        Read a file through FilesystemTools.

        Args:
            file_path: File to read, relative to the workspace root.

        Returns:
            The file's contents.

        Raises:
            ValueError: If `file_path` is empty.
            PathOutsideWorkspaceError: If the path escapes the
                workspace.
            FileTooLargeError: If the file exceeds the size ceiling.
            ToolExecutionError: If the file is missing or unreadable.
        """
        if not file_path or not str(file_path).strip():
            raise ValueError("file_path must be a non-empty string.")
        return self.filesystem.read_file(file_path)

    def _analyze_source(
        self, file_path: str, source: str, include_todos: bool
    ) -> List[BugReport]:
        """
        Run every check against already-read source.

        Args:
            file_path: File being analyzed.
            source: Its contents.
            include_todos: Whether to report TODO markers.

        Returns:
            Findings ordered by line number. A file that will not parse
            yields exactly one syntax-error finding, since no other
            check can run on it.

        Raises:
            StaticAnalysisError: If the file passes the syntax check but
                still will not parse, which should be impossible and so
                is worth surfacing loudly rather than swallowing.
        """
        lines = source.split("\n")

        syntax = self.check_syntax(file_path, source=source)
        if syntax:
            return syntax

        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError) as exc:
            raise StaticAnalysisError(
                f"Could not parse {file_path!r} after it passed the syntax "
                f"check: {exc}"
            ) from exc

        scopes = self._function_index(tree)

        findings: List[BugReport] = []
        findings.extend(self._pyflakes_findings(file_path, tree, lines, scopes))
        findings.extend(self._unreachable(file_path, tree, lines, scopes))
        findings.extend(self._duplicate_definitions(file_path, tree, lines, scopes))
        findings.extend(self._argument_counts(file_path, tree, lines, scopes))
        findings.extend(self._mutable_defaults(file_path, tree, lines, scopes))
        findings.extend(self._bare_excepts(file_path, tree, lines, scopes))
        if include_todos:
            findings.extend(self._todo_markers(file_path, lines, scopes))

        return self._finalize(findings)

    def _run_one(self, file_path: str, check: Any) -> List[BugReport]:
        """
        Read, parse, and run a single AST-based check.

        Backs the public `check_*` methods, which callers may use in
        isolation.

        Args:
            file_path: File to check, relative to the workspace root.
            check: The bound internal check to run.

        Returns:
            That check's findings, or a syntax finding if the file will
            not parse.
        """
        source = self._read(file_path)
        lines = source.split("\n")

        syntax = self.check_syntax(file_path, source=source)
        if syntax:
            return syntax

        tree = ast.parse(source)
        return self._finalize(
            check(file_path, tree, lines, self._function_index(tree))
        )

    def _filtered(self, file_path: str, bug_types: Set[str]) -> List[BugReport]:
        """
        Run pyflakes and keep only the requested bug types.

        Args:
            file_path: File to check, relative to the workspace root.
            bug_types: Bug types to keep.

        Returns:
            The matching findings.
        """
        source = self._read(file_path)
        lines = source.split("\n")

        syntax = self.check_syntax(file_path, source=source)
        if syntax:
            return syntax

        tree = ast.parse(source)
        findings = self._pyflakes_findings(
            file_path, tree, lines, self._function_index(tree)
        )
        return [finding for finding in findings if finding.bug_type in bug_types]

    def _limit_reached(
        self, files_analyzed: int, lines_seen: int
    ) -> Optional[str]:
        """
        Check a run against the proposal's Scope & Limits ceilings.

        Args:
            files_analyzed: Files completed so far.
            lines_seen: Lines read so far.

        Returns:
            A description of the ceiling that has been hit, or None if
            there is room to continue.
        """
        if files_analyzed >= self.config.max_repository_files:
            return (
                f"reached max_repository_files "
                f"({self.config.max_repository_files})"
            )
        if lines_seen >= self.config.max_total_lines_of_code:
            return (
                f"reached max_total_lines_of_code "
                f"({self.config.max_total_lines_of_code})"
            )
        return None

    @staticmethod
    def _statement_blocks(node: ast.AST) -> List[List[ast.stmt]]:
        """
        Collect the statement lists a node owns.

        Args:
            node: Any AST node.

        Returns:
            Each non-empty `body`, `orelse`, or `finalbody` list.
        """
        blocks: List[List[ast.stmt]] = []
        for attribute in ("body", "orelse", "finalbody"):
            block = getattr(node, attribute, None)
            if isinstance(block, list) and block:
                if all(isinstance(item, ast.stmt) for item in block):
                    blocks.append(block)
        return blocks

    @staticmethod
    def _function_index(tree: ast.Module) -> List[Tuple[int, int, str]]:
        """
        Map line ranges to the functions that contain them.

        Args:
            tree: The parsed module.

        Returns:
            Tuples of (first line, last line, qualified name), narrowest
            ranges usable by `_enclosing`.
        """
        index: List[Tuple[int, int, str]] = []

        def walk(node: ast.AST, prefix: Tuple[str, ...]) -> None:
            """Recurse, accumulating the qualified-name prefix."""
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified = ".".join(prefix + (child.name,))
                    index.append(
                        (
                            child.lineno,
                            child.end_lineno or child.lineno,
                            qualified,
                        )
                    )
                    walk(child, prefix + (child.name,))
                elif isinstance(child, ast.ClassDef):
                    walk(child, prefix + (child.name,))
                else:
                    walk(child, prefix)

        walk(tree, ())
        return index

    @staticmethod
    def _enclosing(scopes: List[Tuple[int, int, str]], line: int) -> str:
        """
        Find the innermost function containing a line.

        Args:
            scopes: Ranges from `_function_index`.
            line: The line to attribute.

        Returns:
            The qualified function name, or MODULE_SCOPE when the line
            sits outside every function.
        """
        best: Optional[Tuple[int, str]] = None
        for start, end, name in scopes:
            if start <= line <= end:
                span = end - start
                if best is None or span < best[0]:
                    best = (span, name)
        return best[1] if best else MODULE_SCOPE

    @staticmethod
    def _clamp(line: int, total: int) -> int:
        """
        Keep a line number inside the file.

        pyflakes and SyntaxError occasionally point one line past the
        end, which BugReport's validation would reject.

        Args:
            line: The reported line number.
            total: Number of lines in the file.

        Returns:
            A line number between 1 and `total`.
        """
        return max(1, min(int(line or 1), max(total, 1)))

    def _report(
        self,
        bug_type: str,
        description: str,
        severity: str,
        confidence: float,
        file_path: str,
        function_name: str,
        line_start: int,
        line_end: int,
        lines: List[str],
        suggested_fix: Optional[str] = None,
    ) -> BugReport:
        """
        Assemble a BugReport with evidence quoted from the source.

        Evidence is sliced straight out of `lines` rather than
        described, so GroundingChecker can re-read the file at the same
        range and compare byte for byte. Long spans are truncated to
        `MAX_EVIDENCE_LINES`, and `line_end` is pulled in to match --
        the quote must stay consistent with the range it claims.

        Args:
            bug_type: Category of the finding.
            description: Human-readable explanation.
            severity: "low", "medium", or "high".
            confidence: How likely the finding is a real defect.
            file_path: File the finding is in.
            function_name: Enclosing function, or MODULE_SCOPE.
            line_start: First line of the finding.
            line_end: Last line of the finding.
            lines: The file's source lines.
            suggested_fix: Optional remediation.

        Returns:
            The assembled BugReport.
        """
        start = self._clamp(line_start, len(lines))
        end = self._clamp(max(line_end, start), len(lines))

        if end - start + 1 > MAX_EVIDENCE_LINES:
            end = start + MAX_EVIDENCE_LINES - 1

        return BugReport(
            bug_type=bug_type,
            description=description,
            severity=severity,  # type: ignore[arg-type]
            confidence=confidence,
            file_path=file_path,
            function_name=function_name,
            line_start=start,
            line_end=end,
            evidence="\n".join(lines[start - 1 : end]),
            suggested_fix=suggested_fix,
            detection_method="static",
        )

    @staticmethod
    def _finalize(findings: List[BugReport]) -> List[BugReport]:
        """
        Deduplicate and order findings for stable, readable output.

        Two checks can legitimately catch the same defect -- pyflakes
        reports a redefinition as `RedefinedWhileUnused` while the AST
        pass reports it as a duplicate definition -- and the user should
        see it once. Findings collide when they share a file, a line,
        and a bug type; the more confident one, with its more specific
        message, wins.

        Args:
            findings: Findings to finalize.

        Returns:
            The unique findings, sorted by file, then line, then bug
            type.
        """
        best: Dict[Tuple[str, int, str], BugReport] = {}
        for finding in findings:
            key = (finding.file_path, finding.line_start, finding.bug_type)
            existing = best.get(key)
            if existing is None or finding.confidence > existing.confidence:
                best[key] = finding

        return sorted(
            best.values(),
            key=lambda finding: (
                finding.file_path,
                finding.line_start,
                finding.bug_type,
            ),
        )
