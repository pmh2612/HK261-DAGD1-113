"""Application settings, loaded from the environment and the local `.env` file."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Canonical section names, agreed in `docs/decisions.md` DECIDE-2. PDF splitting emits
#: these and chunk provenance records them, so both sides have to use this one list.
#: "Other" is the catch-all: a heading that maps to nothing else keeps valid provenance
#: instead of having its text dropped.
SECTION_NAMES = (
    "Abstract",
    "Introduction",
    "Related Work",
    "Method",
    "Experiments",
    "Results",
    "Conclusion",
    "Other",
)


class Settings(BaseSettings):
    """Settings for the paper search system.

    Values come from environment variables, falling back to the `.env` file in
    the project root. See `.env.example` for the full list.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    semantic_scholar_api_key: str = ""

    #: Local inference endpoint (Ollama's default). No hosted API is used - see
    #: `docs/decisions.md` DECIDE-7 and `NFR-8`. Nothing leaves this machine.
    llm_base_url: str = "http://localhost:11434"

    #: Open-weights model tag, e.g. "qwen2.5:7b-instruct". Empty until DECIDE-7 picks
    #: one. Re-running extraction under a different model silently produces a different
    #: graph, so this is pinned and recorded in the run manifest.
    llm_model: str = ""

    #: Pinned by `docs/decisions.md` DECIDE-5. Index-time and query-time embeddings must
    #: come from the same model, so changing this invalidates every built index.
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    #: The bge-* models expect this prefix on the QUERY only, never on the documents.
    #: Leaving it off degrades retrieval silently, which is why it is pinned next to the
    #: model instead of living as a literal in the search function.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    data_dir: Path = Field(default=PROJECT_ROOT / "data")

    @property
    def raw_dir(self) -> Path:
        """Directory holding data exactly as downloaded from the sources."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Directory holding normalized, deduplicated and chunked data."""
        return self.data_dir / "processed"


settings = Settings()
