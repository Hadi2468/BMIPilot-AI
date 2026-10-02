"""Application settings, loaded from environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings. Every field can be overridden with a `BMIPILOT_<FIELD>` env var.

    OPENAI_API_KEY and LANGSMITH_* are read directly by their SDKs from the environment.
    """

    model_config = SettingsConfigDict(env_prefix="BMIPILOT_", env_file=".env", extra="ignore")

    # LLM
    openai_model: str = "gpt-4o-mini"
    temperature: float = Field(0.2, ge=0, le=2)
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # Exercise database (API Ninjas)
    ninjas_api_key: SecretStr | None = Field(
        None, validation_alias=AliasChoices("NINJAS_API_KEY", "BMIPILOT_NINJAS_API_KEY")
    )
    exercise_api_url: str = "https://api.api-ninjas.com/v1/exercises"
    exercise_api_timeout_s: float = 10.0
    max_exercises: int = Field(5, ge=1, le=10)  # rows sent back to the LLM

    # Serving
    api_token: str | None = None  # when set, the API requires `Authorization: Bearer <token>`
    api_url: str = "http://localhost:8000"  # used by the Streamlit dashboard


@lru_cache
def get_settings() -> Settings:
    # Export .env into os.environ so the OpenAI / LangSmith SDKs can see their keys.
    load_dotenv()
    return Settings()
