"""Application settings, loaded from the environment and the local `.env` file."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    llm_api_key: str = ""

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
