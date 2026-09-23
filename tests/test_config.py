"""Tests for `paper_search.config`."""

from pathlib import Path

from paper_search.config import Settings, settings


def test_settings_loads():
    assert isinstance(settings, Settings)


def test_neo4j_defaults():
    assert settings.neo4j_uri.startswith(("bolt://", "neo4j://", "neo4j+s://"))
    assert settings.neo4j_user


def test_data_dirs_derive_from_data_dir():
    s = Settings(data_dir=Path("/tmp/example"))
    assert s.raw_dir == Path("/tmp/example/raw")
    assert s.processed_dir == Path("/tmp/example/processed")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("NEO4J_USER", "someone-else")
    assert Settings(_env_file=None).neo4j_user == "someone-else"
