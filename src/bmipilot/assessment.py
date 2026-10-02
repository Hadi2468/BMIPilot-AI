"""Deterministic BMI assessment for every age, with the method that is valid for that age.

| Age group        | Method                                                  |
|------------------|---------------------------------------------------------|
| 0 to < 2 years   | WHO Child Growth Standards: BMI-for-age z-score         |
| 2 to < 20 years  | CDC 2000 Growth Charts: BMI-for-age percentile          |
| 20 to 150 years  | WHO adult BMI classification                            |
"""

from __future__ import annotations

from typing import Any

from bmipilot.growth import (
    DAYS_PER_YEAR,
    WHO_MAX_DAYS,
    Sex,
    lms_at,
    percentile_from_z,
    who_z_score,
)
from bmipilot.policy import ACTIVITY_GUIDELINES
from bmipilot.schemas import ADULT_AGE_YEARS, AgeGroup, Assessment, LifeStage, Profile, RiskLevel

ADULT_NORMAL_BMI = (18.5, 25.0)  # classification cut-offs, upper bound exclusive
ADULT_HEALTHY_RANGE = (18.5, 24.9)  # range shown to users and used for weight targets
WHO_CHILD_NORMAL_Z = (-2.0, 1.0)
CDC_HEALTHY_PERCENTILES = (5.0, 85.0)

AGE_GROUP_LABELS: dict[AgeGroup, str] = {
    "infant_toddler": "0-2 years",
    "child_teen": "2-19 years",
    "adult": "20+ years",
}


def life_stage(age_years: float) -> LifeStage:
    if age_years < 1:
        return "infant"
    if age_years < 3:
        return "toddler"
    if age_years < 5:
        return "preschooler"
    if age_years < 13:
        return "child"
    if age_years < 18:
        return "teen"
    if age_years < 65:
        return "adult"
    return "older_adult"


def age_group(age_years: float) -> AgeGroup:
    if age_years * DAYS_PER_YEAR < WHO_MAX_DAYS:
        return "infant_toddler"
    if age_years < ADULT_AGE_YEARS:
        return "child_teen"
    return "adult"


def classify_adult(bmi: float) -> tuple[str, RiskLevel]:
    """WHO adult classification on the unrounded BMI."""
    if bmi < 16:
        return "Severe thinness", "High"
    if bmi < ADULT_NORMAL_BMI[0]:
        return "Underweight", "Moderate"
    if bmi < ADULT_NORMAL_BMI[1]:
        return "Normal weight", "Low"
    if bmi < 30:
        return "Overweight", "Moderate"
    if bmi < 35:
        return "Obesity (class I)", "High"
    if bmi < 40:
        return "Obesity (class II)", "Very high"
    return "Obesity (class III)", "Very high"


def classify_who_child(z: float) -> tuple[str, RiskLevel]:
    """WHO BMI-for-age cut-offs (z-scores), birth to 2 years."""
    if z < -3:
        return "Severely wasted", "Very high"
    if z < -2:
        return "Wasted", "High"
    if z <= 1:
        return "Normal", "Low"
    if z <= 2:
        return "Possible risk of overweight", "Moderate"
    if z <= 3:
        return "Overweight", "High"
    return "Obese", "Very high"


def classify_cdc_child(percentile: float, pct_of_p95: float) -> tuple[str, RiskLevel]:
    """CDC BMI-for-age categories, 2-19 years (severe obesity as % of the 95th percentile)."""
    if percentile < CDC_HEALTHY_PERCENTILES[0]:
        return "Underweight", "Moderate"
    if percentile < CDC_HEALTHY_PERCENTILES[1]:
        return "Healthy weight", "Low"
    if percentile < 95:
        return "Overweight", "Moderate"
    if pct_of_p95 >= 140:
        return "Severe obesity (class 3)", "Very high"
    if pct_of_p95 >= 120:
        return "Severe obesity (class 2)", "Very high"
    return "Obesity", "High"


def _adult(profile: Profile, bmi: float) -> dict[str, Any]:
    category, risk = classify_adult(bmi)
    low, high = (b * profile.height_m**2 for b in ADULT_HEALTHY_RANGE)
    if bmi < ADULT_NORMAL_BMI[0]:
        to_healthy = low - profile.weight_kg
    elif bmi >= ADULT_NORMAL_BMI[1]:
        to_healthy = high - profile.weight_kg
    else:
        to_healthy = 0.0
    notes = []
    if profile.age_years >= 65:
        notes.append(
            "After 65, a slightly higher BMI may be acceptable; muscle strength and "
            "unintended weight change matter as much as the number."
        )
    return {
        "method": "WHO adult BMI classification",
        "category": category,
        "risk_level": risk,
        "bmi_prime": round(bmi / ADULT_NORMAL_BMI[1], 2),
        "healthy_bmi_min": ADULT_HEALTHY_RANGE[0],
        "healthy_bmi_max": ADULT_HEALTHY_RANGE[1],
        "weight_to_healthy_kg": round(to_healthy, 1),
        "notes": notes,
    }


def _infant(age_days: float, sex: Sex, bmi: float) -> dict[str, Any]:
    lms = lms_at("WHO", sex, age_days)
    z = who_z_score(lms, bmi)
    category, risk = classify_who_child(z)
    return {
        "method": "WHO Child Growth Standards (BMI-for-age z-score)",
        "category": category,
        "risk_level": risk,
        "percentile": round(percentile_from_z(z), 1),
        "z_score": round(z, 2),
        "healthy_bmi_min": round(lms.bmi_at_z(WHO_CHILD_NORMAL_Z[0]), 1),
        "healthy_bmi_max": round(lms.bmi_at_z(WHO_CHILD_NORMAL_Z[1]), 1),
        "notes": ["Under 2 years, BMI should be measured with recumbent length."],
    }


def _child(age_days: float, sex: Sex, bmi: float) -> dict[str, Any]:
    lms = lms_at("CDC", sex, age_days)
    z = lms.z_score(bmi)
    pct_of_p95 = bmi / lms.bmi_at_percentile(95) * 100
    category, risk = classify_cdc_child(percentile_from_z(z), pct_of_p95)
    notes = []
    if pct_of_p95 >= 100:
        notes.append(f"BMI is {pct_of_p95:.0f}% of the 95th percentile for age and sex.")
    return {
        "method": "CDC BMI-for-age percentile",
        "category": category,
        "risk_level": risk,
        "percentile": round(percentile_from_z(z), 1),
        "z_score": round(z, 2),
        "healthy_bmi_min": round(lms.bmi_at_percentile(CDC_HEALTHY_PERCENTILES[0]), 1),
        "healthy_bmi_max": round(lms.bmi_at_percentile(CDC_HEALTHY_PERCENTILES[1]), 1),
        "notes": notes,
    }


def assess(profile: Profile) -> Assessment:
    """Assess a validated profile with the age-appropriate method."""
    bmi = profile.bmi
    group = age_group(profile.age_years)
    stage = life_stage(profile.age_years)
    age_days = profile.age_years * DAYS_PER_YEAR

    if group == "adult":
        result = _adult(profile, bmi)
    else:
        assert profile.sex is not None  # guaranteed by Profile validation under 20
        result = (_infant if group == "infant_toddler" else _child)(age_days, profile.sex, bmi)
        result["notes"] = [
            "Children are not given weight-loss targets; discuss concerns with a paediatrician.",
            *result["notes"],
        ]

    h2 = profile.height_m**2
    return Assessment(
        bmi=round(bmi, 1),
        age_group=group,
        life_stage=stage,
        healthy_weight_min_kg=round(result["healthy_bmi_min"] * h2, 1),
        healthy_weight_max_kg=round(result["healthy_bmi_max"] * h2, 1),
        activity_guideline=ACTIVITY_GUIDELINES[stage],
        **result,
    )


def format_age(age_years: float) -> str:
    days = round(age_years * DAYS_PER_YEAR)
    if days < 31:
        return f"{days} day{'s' if days != 1 else ''}"
    if age_years < 2:
        months = int(age_years * 12)
        return f"{months} month{'s' if months != 1 else ''}"
    years, months = int(age_years), int((age_years % 1) * 12)
    if age_years < 5 and months:
        return f"{years} years {months} months"
    return f"{years} years"


def position_text(a: Assessment) -> str:
    if a.percentile is not None:
        return f"percentile {a.percentile} for age and sex, z-score {a.z_score}"
    return f"BMI Prime {a.bmi_prime}"


def weight_goal_text(a: Assessment) -> str:
    delta = a.weight_to_healthy_kg
    if delta is None:
        return "No weight target for children; focus on healthy growth"
    if delta > 0:
        return f"Gain about {delta} kg to reach the healthy range"
    if delta < 0:
        return f"Lose about {abs(delta)} kg to reach the healthy range"
    return "Within the healthy range"
