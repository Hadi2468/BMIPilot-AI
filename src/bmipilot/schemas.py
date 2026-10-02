"""Input, assessment and output contracts shared by the graph, API, CLI and dashboard."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from bmipilot.growth import Sex

ADULT_AGE_YEARS = 20  # adult cut-offs from 20; younger ages use BMI-for-age charts
PLAUSIBLE_BMI = (7.0, 150.0)  # outside this range the units are almost certainly wrong

LifeStage = Literal["infant", "toddler", "preschooler", "child", "teen", "adult", "older_adult"]
RiskLevel = Literal["Low", "Moderate", "High", "Very high"]
AgeGroup = Literal["infant_toddler", "child_teen", "adult"]


class Profile(BaseModel):
    """A person to assess. Bounds catch unit mistakes, e.g. height 157 (cm) for 1.57 (m)."""

    name: str = Field(min_length=1, max_length=80)
    age_years: float = Field(ge=0, le=150, description="Decimals allowed, e.g. 1.5")
    sex: Sex | None = Field(None, description="Required under 20: growth charts are sex-specific")
    weight_kg: float = Field(ge=0.5, le=400)
    height_m: float = Field(ge=0.3, le=2.6, description="Metres; under 2 use recumbent length")

    @property
    def bmi(self) -> float:
        return self.weight_kg / self.height_m**2

    @model_validator(mode="after")
    def _check_consistency(self) -> Profile:
        if self.age_years < ADULT_AGE_YEARS and self.sex is None:
            raise ValueError("sex is required under age 20 (BMI-for-age charts are sex-specific)")
        if not PLAUSIBLE_BMI[0] <= self.bmi <= PLAUSIBLE_BMI[1]:
            raise ValueError(
                f"BMI {self.bmi:.1f} is implausible; check units (weight in kg, height in metres)"
            )
        return self


class Assessment(BaseModel):
    """Deterministic BMI assessment: computed in code, never by the LLM."""

    bmi: float
    age_group: AgeGroup  # which of the three methods was used
    method: str
    life_stage: LifeStage
    category: str
    risk_level: RiskLevel
    percentile: float | None = None  # children only
    z_score: float | None = None  # children only
    bmi_prime: float | None = None  # adults only: BMI / 25
    healthy_bmi_min: float
    healthy_bmi_max: float
    healthy_weight_min_kg: float
    healthy_weight_max_kg: float
    weight_to_healthy_kg: float | None = None  # adults only; + gain, - lose
    activity_guideline: str
    notes: list[str] = Field(default_factory=list)


class Tip(BaseModel):
    """Structured diet / lifestyle recommendation returned by the LLM."""

    tip: str = Field(description="One practical, actionable tip in at most 25 words.")
    why: str = Field(description="Why it helps for this person, in at most 20 words.")


class CoachResult(BaseModel):
    """Everything a full agent run produces."""

    profile: Profile
    assessment: Assessment
    exercise_plan: str
    exercises: list[dict[str, Any]]
    diet_tip: Tip
    lifestyle_tip: Tip
    explanation: str
    report: str
    warnings: list[str]
    model: str
