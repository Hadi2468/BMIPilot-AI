import pytest

from bmipilot.policy import DISCLAIMER, apply_safety_policy
from bmipilot.service import BMIPilotService
from tests.conftest import FakeLLM, FakeSearch, profile

RISKY = {"exercise_type": "plyometrics", "difficulty": "expert"}
SAFE = {"exercise_type": "cardio", "difficulty": "beginner"}


@pytest.mark.parametrize(
    "stage,risk,expected",
    [
        ("adult", "Low", RISKY),
        ("adult", "High", SAFE),
        ("child", "Low", SAFE),
        ("teen", "Low", SAFE),
        ("older_adult", "Low", SAFE),
    ],
)
def test_safety_policy(stage, risk, expected):
    assert apply_safety_policy(RISKY, stage, risk) == expected


def run(make_graph, settings, llm=None, search=None, **overrides):
    service = BMIPilotService(make_graph(llm, search), settings)
    return service.coach(profile(**overrides))


def test_adult_run_produces_full_report(make_graph, settings):
    llm, search = FakeLLM(), FakeSearch()
    result = run(make_graph, settings, llm, search)
    assert llm.tool_binds == ["get_exercise_plan"]  # tool call is forced
    assert search.queries == [("cardio", "beginner")]
    assert len(result.exercises) == 5  # trimmed to max_exercises
    assert "instructions" not in result.exercises[0]  # trimmed payload
    assert result.diet_tip.tip and result.lifestyle_tip.tip and result.explanation
    assert DISCLAIMER in result.report and result.warnings == []
    assert result.assessment.age_group == "adult"


def test_high_risk_adult_gets_safe_exercise_whatever_the_llm_asks(make_graph, settings):
    search = FakeSearch()
    result = run(make_graph, settings, FakeLLM(RISKY), search, weight_kg=120, height_m=1.75)
    assert result.assessment.risk_level == "Very high"
    assert search.queries == [("cardio", "beginner")]


def test_child_gets_safe_exercise_and_family_guidance(make_graph, settings):
    llm, search = FakeLLM(RISKY), FakeSearch()
    result = run(
        make_graph, settings, llm, search, age_years=9, sex="male", weight_kg=30, height_m=1.35
    )
    assert result.assessment.age_group == "child_teen"
    assert search.queries == [("cardio", "beginner")]
    assert any("Never suggest weight-loss diets" in p for p in llm.prompts)


def test_infant_takes_play_branch_without_tool(make_graph, settings):
    llm, search = FakeLLM(), FakeSearch()
    result = run(
        make_graph,
        settings,
        llm,
        search,
        age_years=1 / 365.25,
        sex="female",
        weight_kg=3.3,
        height_m=0.5,
    )
    assert llm.tool_binds == [] and search.queries == []
    assert result.exercises == [] and "tummy time" in result.exercise_plan
    assert result.assessment.age_group == "infant_toddler"
    assert any("parent or caregiver" in p for p in llm.prompts)


def test_exercise_api_failure_degrades_gracefully(make_graph, settings):
    result = run(make_graph, settings, search=FakeSearch(fail=True))
    assert result.exercises == []
    assert len(result.warnings) == 1 and "## Warnings" in result.report


def test_invalid_tool_arguments_become_a_warning(make_graph, settings):
    llm = FakeLLM({"exercise_type": "yoga", "difficulty": "beginner"})
    result = run(make_graph, settings, llm)
    assert "Invalid tool arguments" in result.warnings[0]


def test_graph_has_parallel_branches(make_graph):
    graph = make_graph().get_graph()
    targets = {e.target for e in graph.edges if e.source == "assess"}
    assert targets == {"exercise_planner", "activity_guide", "diet_planner", "lifestyle_planner"}
