"""
Focused tests: CLI / Config / Streamlit-worker share one Chroma base.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from codebase_assistant.config import (
    Config,
    default_chroma_persist_directory,
    default_runtime_data_dir,
)
from codebase_assistant.rag.store_paths import vector_store_for_repository

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_APP_DIR = _PROJECT_ROOT / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ui_paths import chroma_persist_dir  # noqa: E402


@pytest.fixture
def clean_chroma_env(monkeypatch: pytest.MonkeyPatch):
    """Unset path overrides so defaults are exercised."""
    monkeypatch.delenv("CHROMA_PERSIST_DIR", raising=False)
    monkeypatch.delenv("CODEBASE_ASSISTANT_DATA_DIR", raising=False)
    return monkeypatch


def test_canonical_default_matches_streamlit_temp_layout(clean_chroma_env) -> None:
    expected = os.path.join(
        tempfile.gettempdir(), "codebase_assistant_streamlit", "chroma"
    )
    assert default_chroma_persist_directory() == expected
    assert chroma_persist_dir() == expected
    assert Config().chroma_persist_directory == expected
    assert Config.load().chroma_persist_directory == expected


def test_ui_paths_and_config_load_share_absolute_base(clean_chroma_env) -> None:
    worker_base = os.path.abspath(chroma_persist_dir())
    # Mimic worker.py setdefault then Config.load (Streamlit/worker path).
    clean_chroma_env.setenv("CHROMA_PERSIST_DIR", chroma_persist_dir())
    loaded = Config.load()
    assert os.path.abspath(loaded.chroma_persist_directory) == worker_base


def test_cli_setdefault_matches_worker_base(clean_chroma_env) -> None:
    """CLI main.py setdefault(CHROMA_PERSIST_DIR, chroma_persist_dir())."""
    os.environ.setdefault("CHROMA_PERSIST_DIR", chroma_persist_dir())
    cli_base = os.path.abspath(os.environ["CHROMA_PERSIST_DIR"])
    worker_base = os.path.abspath(chroma_persist_dir())
    assert cli_base == worker_base
    assert os.path.abspath(Config.load().chroma_persist_directory) == worker_base


def test_explicit_chroma_persist_dir_override_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = str(tmp_path / "custom_chroma")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", override)
    monkeypatch.setenv("CODEBASE_ASSISTANT_DATA_DIR", str(tmp_path / "ignored_data"))
    assert default_chroma_persist_directory() == override
    assert chroma_persist_dir() == override
    assert Config().chroma_persist_directory == override
    assert Config.load().chroma_persist_directory == override


def test_data_dir_override_without_chroma_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = tmp_path / "data_root"
    monkeypatch.delenv("CHROMA_PERSIST_DIR", raising=False)
    monkeypatch.setenv("CODEBASE_ASSISTANT_DATA_DIR", str(data))
    expected = os.path.join(str(data), "chroma")
    assert default_runtime_data_dir() == str(data)
    assert default_chroma_persist_directory() == expected
    assert chroma_persist_dir() == expected
    assert Config.load().chroma_persist_directory == expected


def test_same_repo_same_full_store_from_worker_and_cli_config(
    clean_chroma_env, tmp_path: Path
) -> None:
    repo = tmp_path / "medium_like_repo"
    repo.mkdir()
    repo_path = str(repo.resolve())

    # Worker-style resolution
    clean_chroma_env.setenv("CHROMA_PERSIST_DIR", chroma_persist_dir())
    worker_cfg = Config.load()
    worker_store = vector_store_for_repository(
        worker_cfg.chroma_persist_directory, repo_path
    )

    # Fresh CLI-style: clear in-process override simulation by re-resolving
    # through the same canonical helper Config.load uses when env is set
    # the way main.py setdefaults it.
    cli_env = chroma_persist_dir()
    clean_chroma_env.setenv("CHROMA_PERSIST_DIR", cli_env)
    cli_cfg = Config.load()
    cli_store = vector_store_for_repository(
        cli_cfg.chroma_persist_directory, repo_path
    )

    assert os.path.abspath(worker_cfg.chroma_persist_directory) == os.path.abspath(
        cli_cfg.chroma_persist_directory
    )
    assert worker_store == cli_store
    assert worker_store.startswith(
        os.path.abspath(worker_cfg.chroma_persist_directory)
    )
