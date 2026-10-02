"""Markdown report shared by the graph, CLI, API and dashboard."""

from __future__ import annotations

from bmipilot.assessment import (
    AGE_GROUP_LABELS,
    format_age,
    position_text,
    weight_goal_text,
)
from bmipilot.policy import DISCLAIMER
from bmipilot.schemas import Assessment, Profile, Tip


def format_tip(tip: Tip) -> str:
    return f"{tip.tip} *({tip.why})*"


def metric_rows(profile: Profile, a: Assessment) -> list[tuple[str, str]]:
    return [
        ("Age", f"{format_age(profile.age_years)} ({a.life_stage.replace('_', ' ')})"),
        ("Sex", profile.sex or "-"),
        ("Weight", f"{profile.weight_kg} kg"),
        ("Height", f"{profile.height_m} m"),
        ("BMI", str(a.bmi)),
        ("Age group", AGE_GROUP_LABELS[a.age_group]),
        ("Method", a.method),
        ("Position", position_text(a)),
        ("Category", a.category),
        ("Risk level", a.risk_level),
        ("Healthy BMI", f"{a.healthy_bmi_min}-{a.healthy_bmi_max}"),
        ("Healthy weight", f"{a.healthy_weight_min_kg}-{a.healthy_weight_max_kg} kg"),
        ("Weight goal", weight_goal_text(a)),
    ]


def build_report(
    profile: Profile,
    a: Assessment,
    exercise_plan: str,
    diet_tip: Tip,
    lifestyle_tip: Tip,
    explanation: str,
    warnings: list[str],
) -> str:
    lines = [f"# Health Report for {profile.name}", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {label} | {value} |" for label, value in metric_rows(profile, a)]
    lines += [
        "",
        "## Exercise Plan",
        exercise_plan,
        "",
        "## Diet Tip",
        format_tip(diet_tip),
        "",
        "## Lifestyle Tip",
        format_tip(lifestyle_tip),
        "",
        "## Explanation",
        explanation,
    ]
    if a.notes:
        lines += ["", "## Notes", *(f"- {n}" for n in a.notes)]
    if warnings:
        lines += ["", "## Warnings", *(f"- {w}" for w in warnings)]
    lines += ["", "---", f"*{DISCLAIMER}*"]
    return "\n".join(lines)
