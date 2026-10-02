"""Safety policy and public-health guidance, enforced in code rather than by prompt alone."""

from __future__ import annotations

from typing import Any, Literal

from bmipilot.schemas import LifeStage

ExerciseType = Literal["cardio", "stretching", "strength", "plyometrics"]
Difficulty = Literal["beginner", "intermediate", "expert"]

HIGH_RISK_LEVELS = frozenset({"High", "Very high"})
LOW_IMPACT_TYPES: tuple[ExerciseType, ...] = ("cardio", "stretching")
UNDER_FIVE: frozenset[LifeStage] = frozenset({"infant", "toddler", "preschooler"})
CAUTIOUS_STAGES: frozenset[LifeStage] = frozenset({"child", "teen", "older_adult"})

DISCLAIMER = (
    "This report is for general wellness information only and is not medical advice. "
    "BMI does not account for muscle mass, body composition, pregnancy or medical "
    "conditions. Please consult a healthcare professional before starting a new "
    "exercise or diet programme."
)

# WHO guidelines on physical activity: under-5s (2019), everyone 5+ (2020).
_SCHOOL_AGE_ACTIVITY = (
    "An average of 60 minutes a day of moderate-to-vigorous activity, mostly aerobic, "
    "with vigorous and muscle- and bone-strengthening activity on at least 3 days a week."
)
ACTIVITY_GUIDELINES: dict[LifeStage, str] = {
    "infant": (
        "Be physically active several times a day through interactive floor-based play, "
        "including at least 30 minutes of tummy time spread through the day while awake."
    ),
    "toddler": (
        "At least 180 minutes a day of varied physical activity at any intensity, "
        "spread through the day; avoid being restrained for more than 1 hour at a time."
    ),
    "preschooler": (
        "At least 180 minutes a day of varied physical activity, of which at least "
        "60 minutes is moderate-to-vigorous, spread through the day."
    ),
    "child": _SCHOOL_AGE_ACTIVITY,
    "teen": _SCHOOL_AGE_ACTIVITY,
    "adult": (
        "150-300 minutes of moderate or 75-150 minutes of vigorous aerobic activity a week, "
        "plus muscle-strengthening activity on 2 or more days."
    ),
    "older_adult": (
        "150-300 minutes of moderate aerobic activity a week, muscle-strengthening on 2 or "
        "more days, and balance and strength training on 3 or more days to prevent falls."
    ),
}

# Audience and guardrails injected into every prompt.
_YOUNG_CHILD_GUIDANCE = (
    "Write for the parent or caregiver. Never suggest dieting or restricting food; "
    "focus on family meals, active play and routines, and refer to a paediatrician."
)
STAGE_GUIDANCE: dict[LifeStage, str] = {
    "infant": (
        "Write for the parent or caregiver. Never suggest dieting or restricting food; "
        "focus on responsive feeding (breast milk or formula, and age-appropriate "
        "complementary foods from about 6 months) and refer to a paediatrician."
    ),
    "toddler": _YOUNG_CHILD_GUIDANCE,
    "preschooler": _YOUNG_CHILD_GUIDANCE,
    "child": (
        "Write for the child and their family. Never suggest weight-loss diets or calorie "
        "restriction; focus on fun activities and healthy family habits."
    ),
    "teen": (
        "Write for a teenager and their family. Never suggest weight-loss diets or calorie "
        "restriction; promote a positive body image and healthy habits."
    ),
    "adult": "Write for an adult.",
    "older_adult": (
        "Write for an older adult. Prioritise adequate protein, hydration, muscle strength "
        "and balance; avoid aggressive weight loss."
    ),
}


def play_plan(guideline: str) -> str:
    """Deterministic activity plan for under-5s: play, not exercises."""
    return (
        "No structured exercise at this age: activity should come from play.\n\n"
        f"- **WHO guideline:** {guideline}\n"
        "- Make movement part of daily routines: floor play, crawling, walking, dancing, "
        "playground time.\n"
        "- Limit time spent sitting in prams, car seats or in front of screens."
    )


def apply_safety_policy(args: dict[str, Any], stage: LifeStage, risk: str) -> dict[str, Any]:
    """Children, teens, older adults and high-risk users get beginner, low-impact exercises,
    whatever tool arguments the LLM chose."""
    if stage not in CAUTIOUS_STAGES and risk not in HIGH_RISK_LEVELS:
        return args
    safe = {**args, "difficulty": "beginner"}
    if safe.get("exercise_type") not in LOW_IMPACT_TYPES:
        safe["exercise_type"] = LOW_IMPACT_TYPES[0]
    return safe
