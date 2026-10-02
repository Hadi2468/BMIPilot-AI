"""All prompt templates. Every prompt receives the same profile block, including the
life-stage audience and safety rules from `policy.STAGE_GUIDANCE`."""

from __future__ import annotations

import textwrap
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from bmipilot.assessment import format_age, position_text
from bmipilot.policy import STAGE_GUIDANCE
from bmipilot.schemas import Assessment, Profile


def _dedent(text: str) -> str:
    return textwrap.dedent(text).strip()


PROFILE_TEMPLATE = _dedent(
    """
    Name: {name}
    Age: {age} (life stage: {life_stage})
    Sex: {sex}
    Weight: {weight_kg} kg
    Height: {height_m} m
    BMI: {bmi}, assessed with: {method}
    Result: {category} ({position}), risk level: {risk_level}
    Healthy weight range for this height: {healthy_weight_min_kg}-{healthy_weight_max_kg} kg
    WHO physical activity guideline for this age: {activity_guideline}

    Audience and safety rules: {stage_guidance}
    """
)


def prompt_vars(profile: Profile, a: Assessment) -> dict[str, Any]:
    return {
        "name": profile.name,
        "age": format_age(profile.age_years),
        "life_stage": a.life_stage.replace("_", " "),
        "sex": profile.sex or "not given",
        "weight_kg": profile.weight_kg,
        "height_m": profile.height_m,
        "bmi": a.bmi,
        "method": a.method,
        "category": a.category,
        "position": position_text(a),
        "risk_level": a.risk_level,
        "healthy_weight_min_kg": a.healthy_weight_min_kg,
        "healthy_weight_max_kg": a.healthy_weight_max_kg,
        "activity_guideline": a.activity_guideline,
        "stage_guidance": STAGE_GUIDANCE[a.life_stage],
    }


EXERCISE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            _dedent(
                """
                You are an exercise planning assistant.
                1. Based on the user's age, BMI result and risk level, choose an exercise
                   type and difficulty and call the get_exercise_plan tool.
                   Prefer low-impact, beginner exercises for children, older adults and
                   higher risk levels.
                2. Using the tool results, recommend 2-3 exercises with frequency and duration
                   that fit the WHO activity guideline, as short Markdown bullets
                   (at most 80 words). For children, make them playful.
                Only recommend tool results that are safe for this person. For older adults
                and higher risk levels, avoid jumping, kicking, sliding and floor-to-standing
                moves; if no result is suitable, say so briefly and recommend brisk walking
                or chair-based exercises instead. Only if the tool reports an error, say the
                exercise database was unavailable and suggest a generic safe activity.
                """
            ),
        ),
        ("human", PROFILE_TEMPLATE),
    ]
)

DIET_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            _dedent(
                """
                You are a nutrition planning assistant.
                Give one short, practical diet tip based on the user's age and BMI result.
                The tip must be about food, drinks or eating habits, never about exercise.
                Follow the audience and safety rules exactly.
                Do not give calorie targets and do not diagnose medical conditions.
                """
            ),
        ),
        ("human", PROFILE_TEMPLATE),
    ]
)

LIFESTYLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            _dedent(
                """
                You are a lifestyle coaching assistant.
                Give one short, practical lifestyle tip based on the user's age and BMI result,
                about sleep, hydration, screen time, stress management or daily routines.
                Follow the audience and safety rules exactly. Do not repeat diet or exercise
                advice and do not diagnose medical conditions.
                """
            ),
        ),
        ("human", PROFILE_TEMPLATE),
    ]
)

EXPLANATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            _dedent(
                """
                You are a health explanation assistant.
                In at most 3 short sentences, explain in plain language what the BMI result
                means for this person's age and how it connects to the recommendations.
                Be encouraging. Follow the audience and safety rules exactly.
                Do not diagnose medical conditions.
                """
            ),
        ),
        (
            "human",
            _dedent(
                """
                {profile}

                Exercise recommendation:
                {exercise_plan}

                Diet recommendation: {diet_tip}

                Lifestyle recommendation: {lifestyle_tip}
                """
            ),
        ),
    ]
)
