"""LangSmith tracing and per-run config.

Tracing is enabled purely through environment variables (LANGSMITH_TRACING=true,
LANGSMITH_API_KEY, LANGSMITH_PROJECT); LangChain/LangGraph pick them up automatically.
Here we only attach a run name, tags, and metadata so traces are easy to filter.
"""

from __future__ import annotations

import logging
import os

from langchain_core.runnables import RunnableConfig

from bmipilot import __version__
from bmipilot.assessment import age_group, life_stage
from bmipilot.config import Settings
from bmipilot.schemas import Profile

logger = logging.getLogger(__name__)


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(
        os.getenv("LANGSMITH_API_KEY")
    )


def log_tracing_status() -> None:
    if tracing_enabled():
        logger.info("LangSmith tracing ON (project=%s)", os.getenv("LANGSMITH_PROJECT", "default"))
    else:
        logger.info("LangSmith tracing OFF")


def run_config(profile: Profile, settings: Settings) -> RunnableConfig:
    group = age_group(profile.age_years)
    return {
        "run_name": "bmipilot",
        "tags": ["bmipilot", settings.openai_model, group],
        "metadata": {
            "app_version": __version__,
            "model": settings.openai_model,
            "age_group": group,
            "life_stage": life_stage(profile.age_years),
        },
    }
