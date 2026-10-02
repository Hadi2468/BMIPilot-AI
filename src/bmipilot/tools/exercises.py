"""Exercise search tool. The provider (API Ninjas) is injectable, so tests run offline."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

import requests
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bmipilot.policy import Difficulty, ExerciseType

logger = logging.getLogger(__name__)

ExerciseSearchFn = Callable[[str, str], Sequence[dict[str, Any]]]

TOOL_NAME = "get_exercise_plan"
RESULT_FIELDS = ("name", "type", "muscle", "equipment", "difficulty")


class ExerciseQuery(BaseModel):
    exercise_type: ExerciseType = Field(description="cardio, stretching, strength or plyometrics")
    difficulty: Difficulty = Field(description="beginner, intermediate or expert")


def make_ninjas_search(api_key: str | None, url: str, timeout_s: float) -> ExerciseSearchFn:
    """API Ninjas client with retries. Successful results are cached: there are only a few
    type/difficulty pairs, so repeated runs rarely hit the network."""
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))

    @lru_cache(maxsize=32)
    def search(exercise_type: str, difficulty: str) -> tuple[dict[str, Any], ...]:
        if not api_key:
            raise RuntimeError("NINJAS_API_KEY is missing")
        response = session.get(
            url,
            headers={"X-Api-Key": api_key},
            params={"type": exercise_type, "difficulty": difficulty},
            timeout=timeout_s,
        )
        response.raise_for_status()
        return tuple(response.json())

    return search


def make_exercise_tool(search_fn: ExerciseSearchFn, max_results: int = 5) -> BaseTool:
    """Wrap a search function as an LLM tool with enum-typed arguments and trimmed output."""

    def get_exercise_plan(exercise_type: ExerciseType, difficulty: Difficulty) -> list[dict]:
        try:
            rows = search_fn(exercise_type, difficulty)
        except Exception as exc:  # any provider failure degrades gracefully
            logger.warning("Exercise search failed: %s", exc)
            return [{"error": f"Exercise database unavailable ({type(exc).__name__})."}]
        return [{k: row.get(k) for k in RESULT_FIELDS} for row in list(rows)[:max_results]]

    return StructuredTool.from_function(
        func=get_exercise_plan,
        name=TOOL_NAME,
        description=(
            "Search the exercise database. Returns up to "
            f"{max_results} exercises (name, type, muscle, equipment, difficulty), "
            'or one item with an "error" key if the database is unavailable.'
        ),
        args_schema=ExerciseQuery,
    )
