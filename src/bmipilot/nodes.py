"""Graph nodes. The LLM and the exercise tool are injected, so the graph runs offline in tests."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool

from bmipilot.assessment import assess
from bmipilot.policy import UNDER_FIVE, apply_safety_policy, play_plan
from bmipilot.prompts import (
    DIET_PROMPT,
    EXERCISE_PROMPT,
    EXPLANATION_PROMPT,
    LIFESTYLE_PROMPT,
    PROFILE_TEMPLATE,
    prompt_vars,
)
from bmipilot.report import build_report, format_tip
from bmipilot.schemas import Assessment, Profile, Tip
from bmipilot.state import GraphState


def _load(state: GraphState) -> tuple[Profile, Assessment]:
    return Profile(**state["profile"]), Assessment(**state["assessment"])


def route_activity(state: GraphState) -> str:
    """Under-5s get play-based guidance; everyone else gets the tool-using planner."""
    if state["assessment"]["life_stage"] in UNDER_FIVE:
        return "activity_guide"
    return "exercise_planner"


class Nodes:
    def __init__(self, llm: BaseChatModel, exercise_tool: BaseTool):
        self.llm = llm
        self.exercise_tool = exercise_tool

    # ------------------------------------------------------------
    # Deterministic nodes
    # ------------------------------------------------------------

    def validate(self, state: GraphState) -> dict[str, Any]:
        return {"profile": Profile(**state["profile"]).model_dump()}

    def assess(self, state: GraphState) -> dict[str, Any]:
        return {"assessment": assess(Profile(**state["profile"])).model_dump()}

    def activity_guide(self, state: GraphState) -> dict[str, Any]:
        guideline = state["assessment"]["activity_guideline"]
        return {"exercise_plan": play_plan(guideline), "exercises": []}

    def report(self, state: GraphState) -> dict[str, Any]:
        profile, a = _load(state)
        report = build_report(
            profile,
            a,
            state["exercise_plan"],
            Tip(**state["diet_tip"]),
            Tip(**state["lifestyle_tip"]),
            state["explanation"],
            state.get("warnings", []),
        )
        return {"report": report}

    # ------------------------------------------------------------
    # LLM nodes
    # ------------------------------------------------------------

    def exercise_planner(self, state: GraphState) -> dict[str, Any]:
        profile, a = _load(state)
        messages = EXERCISE_PROMPT.format_messages(**prompt_vars(profile, a))

        # Step 1: the LLM must pick tool arguments (tool_choice forces a tool call)
        tool = self.exercise_tool
        ai_msg = self.llm.bind_tools([tool], tool_choice=tool.name).invoke(messages)
        messages.append(ai_msg)

        # Step 2: run every tool call, with the safety policy applied to its arguments
        exercises: list[dict[str, Any]] = []
        warnings: list[str] = []
        for call in ai_msg.tool_calls:
            if call["name"] != tool.name:
                result = [{"error": f"Unknown tool: {call['name']}"}]
            else:
                safe_args = apply_safety_policy(call["args"], a.life_stage, a.risk_level)
                try:
                    result = tool.invoke(safe_args)
                except ValueError as exc:  # invalid arguments from the LLM
                    result = [{"error": f"Invalid tool arguments: {exc}"}]
            warnings.extend(row["error"] for row in result if "error" in row)
            exercises.extend(row for row in result if "error" not in row)
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))

        # Step 3: the LLM writes the final recommendation from the tool results
        final_msg = self.llm.invoke(messages)

        update: dict[str, Any] = {"exercise_plan": final_msg.content, "exercises": exercises}
        if warnings:
            update["warnings"] = warnings
        return update

    def diet_planner(self, state: GraphState) -> dict[str, Any]:
        tip = (DIET_PROMPT | self.llm.with_structured_output(Tip)).invoke(
            prompt_vars(*_load(state))
        )
        return {"diet_tip": tip.model_dump()}

    def lifestyle_planner(self, state: GraphState) -> dict[str, Any]:
        tip = (LIFESTYLE_PROMPT | self.llm.with_structured_output(Tip)).invoke(
            prompt_vars(*_load(state))
        )
        return {"lifestyle_tip": tip.model_dump()}

    def explain(self, state: GraphState) -> dict[str, Any]:
        chain = EXPLANATION_PROMPT | self.llm | StrOutputParser()
        explanation = chain.invoke(
            {
                "profile": PROFILE_TEMPLATE.format(**prompt_vars(*_load(state))),
                "exercise_plan": state["exercise_plan"],
                "diet_tip": format_tip(Tip(**state["diet_tip"])),
                "lifestyle_tip": format_tip(Tip(**state["lifestyle_tip"])),
            }
        )
        return {"explanation": explanation}
