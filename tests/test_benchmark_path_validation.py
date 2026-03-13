import os

import pytest

from thunder.benchmark import _ensure_path_component, _resolve_base_embeddings_folder


def test_ensure_path_component_rejects_none():
    with pytest.raises(ValueError, match="cfg.task.base_embeddings_folder"):
        _ensure_path_component("cfg.task.base_embeddings_folder", None)


def test_resolve_base_embeddings_folder_prefers_explicit_value(monkeypatch):
    monkeypatch.setenv("THUNDER_BASE_DATA_FOLDER", "/tmp/base")
    assert _resolve_base_embeddings_folder("/custom/embeddings") == "/custom/embeddings"


def test_resolve_base_embeddings_folder_uses_env(monkeypatch):
    monkeypatch.setenv("THUNDER_BASE_DATA_FOLDER", "/tmp/base")
    resolved = _resolve_base_embeddings_folder(None)
    assert resolved == os.path.join("/tmp/base", "embeddings")


def test_resolve_base_embeddings_folder_fails_without_config_or_env(monkeypatch):
    monkeypatch.delenv("THUNDER_BASE_DATA_FOLDER", raising=False)
    with pytest.raises(ValueError, match="THUNDER_BASE_DATA_FOLDER"):
        _resolve_base_embeddings_folder(None)
