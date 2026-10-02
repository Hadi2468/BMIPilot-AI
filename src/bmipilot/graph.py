"""Graph assembly. Dependencies are injected so the graph is testable without API keys."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from bmipilot.config import Settings, get_settings
from bmipilot.nodes import Nodes, route_activity
from bmipilot.state import GraphState
from bmipilot.tools.exercises import ExerciseSearchFn, make_exercise_tool

ACTIVITY_BRANCHES = ("exercise_planner", "activity_guide")


def build_graph(
    llm: BaseChatModel,
    exercise_search: ExerciseSearchFn,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """
                               ┌→ exercise_planner (tool) ─┐   age 5+
                               ├→ activity_guide ──────────┤   under 5 (play, no tool)
    START → validate → assess ─┼→ diet_planner ────────────┼→ explain → report → END
                               └→ lifestyle_planner ───────┘
    """
    settings = settings or get_settings()
    nodes = Nodes(llm, make_exercise_tool(exercise_search, settings.max_exercises))

    builder = StateGraph(GraphState)
    builder.add_node("validate", nodes.validate)
    builder.add_node("assess", nodes.assess)
    builder.add_node("exercise_planner", nodes.exercise_planner)
    builder.add_node("activity_guide", nodes.activity_guide)
    builder.add_node("diet_planner", nodes.diet_planner)
    builder.add_node("lifestyle_planner", nodes.lifestyle_planner)
    builder.add_node("explain", nodes.explain)
    builder.add_node("report", nodes.report)

    builder.add_edge(START, "validate")
    builder.add_edge("validate", "assess")

    # Fan out: one activity branch (chosen by age) runs in parallel with diet and lifestyle
    builder.add_conditional_edges("assess", route_activity, list(ACTIVITY_BRANCHES))
    builder.add_edge("assess", "diet_planner")
    builder.add_edge("assess", "lifestyle_planner")

    # Fan in: explain runs once, after every branch of the parallel step has finished
    for branch in (*ACTIVITY_BRANCHES, "diet_planner", "lifestyle_planner"):
        builder.add_edge(branch, "explain")

    builder.add_edge("explain", "report")
    builder.add_edge("report", END)
    return builder.compile()


def build_default_graph(settings: Settings | None = None) -> CompiledStateGraph:
    """Wire the production dependencies: OpenAI and the API Ninjas exercise database."""
    from bmipilot.llm import build_llm
    from bmipilot.tools.exercises import make_ninjas_search

    settings = settings or get_settings()
    key = settings.ninjas_api_key.get_secret_value() if settings.ninjas_api_key else None
    search = make_ninjas_search(key, settings.exercise_api_url, settings.exercise_api_timeout_s)
    return build_graph(build_llm(settings), search, settings)
