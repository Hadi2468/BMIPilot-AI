"""Service layer: the single entry point used by the CLI and the FastAPI app (and, through
the API, the Streamlit dashboard), so all interfaces behave identically."""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from bmipilot.assessment import assess
from bmipilot.config import Settings, get_settings
from bmipilot.observability import run_config
from bmipilot.schemas import Assessment, CoachResult, Profile, Tip


class BMIPilotService:
    def __init__(self, graph: CompiledStateGraph, settings: Settings | None = None):
        self.graph = graph
        self.settings = settings or get_settings()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> BMIPilotService:
        from bmipilot.graph import build_default_graph

        settings = settings or get_settings()
        return cls(build_default_graph(settings), settings)

    @staticmethod
    def assess(profile: Profile) -> Assessment:
        """Deterministic assessment only: instant, free, no LLM."""
        return assess(profile)

    def coach(self, profile: Profile) -> CoachResult:
        """Full agent run: assessment + exercise, diet and lifestyle plan + explanation."""
        state = self.graph.invoke(
            {"profile": profile.model_dump(), "warnings": []},
            run_config(profile, self.settings),
        )
        return CoachResult(
            profile=Profile(**state["profile"]),
            assessment=Assessment(**state["assessment"]),
            exercise_plan=state["exercise_plan"],
            exercises=state["exercises"],
            diet_tip=Tip(**state["diet_tip"]),
            lifestyle_tip=Tip(**state["lifestyle_tip"]),
            explanation=state["explanation"],
            report=state["report"],
            warnings=state.get("warnings", []),
            model=self.settings.openai_model,
        )
