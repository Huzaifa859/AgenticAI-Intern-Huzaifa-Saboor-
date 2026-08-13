"""
test_github_clone_cache.py
==========================

Focused tests for durable multi-repository GitHub clone persistence.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

_APP_DIR = Path(__file__).resolve().parents[2] / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import service as app_service  # noqa: E402
from ui_paths import durable_github_clone_path, github_clones_dir  # noqa: E402
from codebase_assistant.rag.store_paths import vector_store_for_repository  # noqa: E402


@pytest.fixture()
def clones_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "github_clones"
    root.mkdir()
    monkeypatch.setenv("GITHUB_CLONES_DIR", str(root))
    monkeypatch.setenv("CODEBASE_ASSISTANT_DATA_DIR", str(tmp_path / "data"))
    # Reset in-process cache between tests.
    app_service._CLONE_CACHE.clear()
    app_service._TEMPORARY_ROOTS.clear()
    return root


def _fake_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".git").mkdir(exist_ok=True)
    (path / "README.md").write_text("ok\n", encoding="utf-8")


def _tools_that_clone() -> MagicMock:
    tools = MagicMock()
    tools.validate_repository.return_value = True

    def _clone(url: str, destination: str = ".") -> bool:
        _fake_git_repo(Path(destination))
        return True

    tools.clone_repository.side_effect = _clone
    return tools


def test_same_url_same_durable_path(clones_root: Path) -> None:
    url = "https://github.com/acme/demo"
    a = durable_github_clone_path(url, clones_root=str(clones_root))
    b = durable_github_clone_path(
        "https://github.com/Acme/Demo.git", clones_root=str(clones_root)
    )
    assert a == b
    assert a == os.path.join(str(clones_root), "github.com", "acme", "demo")


def test_different_url_different_durable_path(clones_root: Path) -> None:
    a = durable_github_clone_path(
        "https://github.com/acme/one", clones_root=str(clones_root)
    )
    b = durable_github_clone_path(
        "https://github.com/acme/two", clones_root=str(clones_root)
    )
    assert a != b


def test_first_load_clones_repository(clones_root: Path) -> None:
    tools = _tools_that_clone()
    path = app_service.clone_or_reuse_repository(
        tools, "https://github.com/acme/demo"
    )
    assert path == durable_github_clone_path(
        "https://github.com/acme/demo", clones_root=str(clones_root)
    )
    assert (Path(path) / ".git").is_dir()
    assert tools.clone_repository.call_count == 1


def test_second_load_same_process_reuses_clone(clones_root: Path) -> None:
    tools = _tools_that_clone()
    url = "https://github.com/acme/demo"
    first = app_service.clone_or_reuse_repository(tools, url)
    second = app_service.clone_or_reuse_repository(tools, url)
    assert first == second
    assert tools.clone_repository.call_count == 1


def test_fresh_process_cache_empty_reuses_durable_clone(
    clones_root: Path,
) -> None:
    tools = _tools_that_clone()
    url = "https://github.com/acme/demo"
    first = app_service.clone_or_reuse_repository(tools, url)
    # Simulate fresh process: empty in-memory cache only.
    app_service._CLONE_CACHE.clear()
    tools2 = _tools_that_clone()
    second = app_service.clone_or_reuse_repository(tools2, url)
    assert second == first
    assert tools2.clone_repository.call_count == 0


def test_cleanup_does_not_delete_durable_clones(clones_root: Path) -> None:
    tools = _tools_that_clone()
    path = app_service.clone_or_reuse_repository(
        tools, "https://github.com/acme/demo"
    )
    # Register a genuine temporary root and ensure cleanup only removes that.
    temp_root = clones_root.parent / "temp_only"
    temp_root.mkdir()
    (temp_root / "marker.txt").write_text("x", encoding="utf-8")
    app_service._TEMPORARY_ROOTS.append(str(temp_root))

    app_service.cleanup_temporary_clones()

    assert Path(path).is_dir()
    assert (Path(path) / ".git").is_dir()
    assert not temp_root.exists()
    assert path not in app_service._TEMPORARY_ROOTS


def test_same_url_same_chroma_identity(clones_root: Path, tmp_path: Path) -> None:
    tools = _tools_that_clone()
    url = "https://github.com/acme/demo"
    path_a = app_service.clone_or_reuse_repository(tools, url)
    app_service._CLONE_CACHE.clear()
    path_b = app_service.clone_or_reuse_repository(tools, url)
    assert path_a == path_b
    chroma = tmp_path / "chroma"
    store_a = vector_store_for_repository(str(chroma), path_a)
    store_b = vector_store_for_repository(str(chroma), path_b)
    assert store_a == store_b


def test_multi_repo_independent_persistence(clones_root: Path) -> None:
    tools = _tools_that_clone()
    urls = [
        "https://github.com/org/repo1",
        "https://github.com/org/repo2",
        "https://github.com/org/repo3",
    ]
    sequence = [
        urls[0],
        urls[1],
        urls[0],
        urls[2],
        urls[1],
        urls[2],
    ]
    paths: List[str] = []
    for url in sequence:
        # Clear in-memory cache between steps to force durable-disk lookup.
        app_service._CLONE_CACHE.clear()
        paths.append(app_service.clone_or_reuse_repository(tools, url))

    p1a, p2a, p1b, p3a, p2b, p3b = paths
    assert p1a == p1b
    assert p2a == p2b
    assert p3a == p3b
    assert len({p1a, p2a, p3a}) == 3
    # Only three real clones for three URLs.
    assert tools.clone_repository.call_count == 3

    app_service.cleanup_temporary_clones()
    assert Path(p1a).is_dir()
    assert Path(p2a).is_dir()
    assert Path(p3a).is_dir()


def test_integration_warm_index_after_fresh_cache(
    clones_root: Path, tmp_path: Path
) -> None:
    """First resolve+index, then empty cache and warm-index again."""
    from codebase_assistant.config import Config
    from codebase_assistant.rag.indexer import Indexer
    from codebase_assistant.rag.ingest import IngestionResult

    tools = MagicMock()
    tools.validate_repository.return_value = True

    def _clone(url: str, destination: str = ".") -> bool:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        (root / ".git").mkdir(exist_ok=True)
        (root / "math_utils.py").write_text(
            "def add(a, b):\n    return a + b\n", encoding="utf-8"
        )
        return True

    tools.clone_repository.side_effect = _clone
    url = "https://github.com/acme/lib"
    chroma = tmp_path / "chroma"

    path1 = app_service.clone_or_reuse_repository(tools, url)
    config = Config(chroma_persist_directory=str(chroma), workspace_root=path1)
    store = vector_store_for_repository(str(chroma), path1)
    indexer = Indexer(
        vector_store_path=store, config=config, workspace_root=path1
    )
    outcome = IngestionResult(files_indexed=1, chunks_indexed=1, lines_indexed=2)
    ingest1 = MagicMock(return_value=outcome)
    indexer.ingestor.ingest_file = ingest1  # type: ignore[method-assign]
    first = indexer.update_index(".")
    assert first.added
    assert ingest1.call_count == 1

    app_service._CLONE_CACHE.clear()
    path2 = app_service.clone_or_reuse_repository(tools, url)
    assert path2 == path1
    assert tools.clone_repository.call_count == 1

    indexer2 = Indexer(
        vector_store_path=vector_store_for_repository(str(chroma), path2),
        config=config,
        workspace_root=path2,
    )
    ingest2 = MagicMock(return_value=outcome)
    indexer2.ingestor.ingest_file = ingest2  # type: ignore[method-assign]
    second = indexer2.update_index(".")
    assert second.added == []
    assert second.modified == []
    assert second.unchanged
    assert ingest2.call_count == 0


def test_github_clones_dir_under_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_CLONES_DIR", raising=False)
    monkeypatch.setenv("CODEBASE_ASSISTANT_DATA_DIR", str(tmp_path / "data"))
    assert github_clones_dir() == str(tmp_path / "data" / "github_clones")
