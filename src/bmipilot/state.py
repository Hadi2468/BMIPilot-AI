"""Graph state. Everything is JSON-serialisable (Pydantic models are stored via model_dump)."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class GraphState(TypedDict, total=False):
    # Input (a validated `Profile`)
    profile: dict[str, Any]
    # Deterministic assessment (an `Assessment`)
    assessment: dict[str, Any]
    # LLM / tool outputs
    exercise_plan: str
    exercises: list[dict[str, Any]]
    diet_tip: dict[str, str]
    lifestyle_tip: dict[str, str]
    explanation: str
    report: str
    # Parallel branches may append warnings in the same step, so they need a reducer
    warnings: Annotated[list[str], operator.add]
