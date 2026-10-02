"""Shared fixtures. Everything runs offline: no API keys, no network, no cost."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from bmipilot.config import Settings
from bmipilot.graph import build_graph
from bmipilot.schemas import Profile, Tip


class FakeLLM(RunnableLambda):
    """Stands in for ChatOpenAI.

    - `bind_tools(...)` returns a model that requests one tool call with `tool_args`.
    - `with_structured_output(Tip)` returns a fixed tip.
    - Invoked directly (or piped), it returns a short text message.
    Every prompt it receives is recorded in `prompts` for assertions.
    """

    model_name = "fake"

    def __init__(self, tool_args: dict[str, Any] | None = None):
        super().__init__(self._text)
        self.tool_args = tool_args or {"exercise_type": "cardio", "difficulty": "beginner"}
        self.tool_binds: list[str | None] = []
        self.prompts: list[str] = []

    def _text(self, prompt: Any) -> AIMessage:
        self.prompts.append(str(prompt))
        return AIMessage(content="Keep up the healthy habits.")

    def bind_tools(self, tools: list, tool_choice: str | None = None) -> RunnableLambda:
        self.tool_binds.append(tool_choice)
        call = {"name": tools[0].name, "args": self.tool_args, "id": "call_1"}
        return RunnableLambda(lambda _: AIMessage(content="", tool_calls=[call]))

    def with_structured_output(self, schema: type) -> RunnableLambda:
        def respond(prompt_value: Any) -> Tip:
            self.prompts.append(prompt_value.to_string())
            return schema(tip="Drink water with every meal.", why="Supports energy.")

        return RunnableLambda(respond)


class FakeSearch:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.queries: list[tuple[str, str]] = []

    def __call__(self, exercise_type: str, difficulty: str) -> list[dict[str, Any]]:
        self.queries.append((exercise_type, difficulty))
        if self.fail:
            raise RuntimeError("exercise provider down")
        return [
            {
                "name": f"Exercise {i}",
                "type": exercise_type,
                "muscle": "quadriceps",
                "equipment": "body_only",
                "difficulty": difficulty,
                "instructions": "x" * 500,
            }
            for i in range(8)
        ]


@pytest.fixture
def settings() -> Settings:
    # _env_file=None: tests must not depend on the developer's local .env
    return Settings(_env_file=None, max_exercises=5)


@pytest.fixture
def make_graph(settings):
    def _make(llm: FakeLLM | None = None, search: FakeSearch | None = None):
        return build_graph(llm or FakeLLM(), search or FakeSearch(), settings)

    return _make


def profile(**overrides: Any) -> Profile:
    base = {"name": "Test", "age_years": 49, "sex": "female", "weight_kg": 62, "height_m": 1.57}
    return Profile(**(base | overrides))
