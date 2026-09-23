"""Tests for `paper_search.config`."""

from pathlib import Path

from paper_search.config import SECTION_NAMES, Settings, settings


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


def test_embedding_model_and_query_prefix_are_pinned():
    # DECIDE-5: changing either of these invalidates every built index, so they are
    # pinned rather than left to a caller.
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_query_prefix.endswith(": ")


def test_section_names_cover_the_agreed_set():
    # DECIDE-2: the shared contract between PDF splitting and chunk provenance.
    assert SECTION_NAMES[0] == "Abstract"
    assert SECTION_NAMES[-1] == "Other"
    assert len(SECTION_NAMES) == len(set(SECTION_NAMES)) == 8
