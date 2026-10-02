import pytest
from pydantic import ValidationError

from bmipilot.assessment import (
    age_group,
    assess,
    classify_adult,
    classify_cdc_child,
    classify_who_child,
    format_age,
    life_stage,
)
from bmipilot.growth import DAYS_PER_YEAR, lms_at
from tests.conftest import profile


@pytest.mark.parametrize(
    "bmi,label,risk",
    [
        (15.9, "Severe thinness", "High"),
        (18.49, "Underweight", "Moderate"),
        (18.5, "Normal weight", "Low"),
        (24.996, "Normal weight", "Low"),  # unrounded BMI, so no false "Overweight"
        (25, "Overweight", "Moderate"),
        (30, "Obesity (class I)", "High"),
        (35, "Obesity (class II)", "Very high"),
        (40, "Obesity (class III)", "Very high"),
    ],
)
def test_classify_adult(bmi, label, risk):
    assert classify_adult(bmi) == (label, risk)


@pytest.mark.parametrize(
    "z,label",
    [
        (-3.1, "Severely wasted"),
        (-2.5, "Wasted"),
        (0, "Normal"),
        (1.5, "Possible risk of overweight"),
        (2.5, "Overweight"),
        (3.5, "Obese"),
    ],
)
def test_classify_who_child(z, label):
    assert classify_who_child(z)[0] == label


@pytest.mark.parametrize(
    "pct,pct_p95,label",
    [
        (3, 70, "Underweight"),
        (50, 85, "Healthy weight"),
        (90, 97, "Overweight"),
        (96, 110, "Obesity"),
        (99, 125, "Severe obesity (class 2)"),
        (99.9, 145, "Severe obesity (class 3)"),
    ],
)
def test_classify_cdc_child(pct, pct_p95, label):
    assert classify_cdc_child(pct, pct_p95)[0] == label


def test_age_groups_switch_at_2_and_20_years():
    assert age_group(1.99) == "infant_toddler"
    assert age_group(2.0) == "child_teen"
    assert age_group(19.99) == "child_teen"
    assert age_group(20) == "adult"
    assert [life_stage(a) for a in (0, 1, 3, 5, 13, 18, 65, 150)] == [
        "infant",
        "toddler",
        "preschooler",
        "child",
        "teen",
        "adult",
        "older_adult",
        "older_adult",
    ]


def test_newborn_at_who_median_is_normal():
    lms = lms_at("WHO", "female", 1)
    a = assess(profile(age_years=1 / DAYS_PER_YEAR, weight_kg=lms.M * 0.25, height_m=0.5))
    assert a.age_group == "infant_toddler" and a.category == "Normal"
    assert a.z_score == pytest.approx(0, abs=0.01)
    assert a.weight_to_healthy_kg is None  # no weight targets for children
    assert any("recumbent length" in n for n in a.notes)


def test_child_categories_follow_cdc_percentiles():
    age, h = 9.0, 1.35
    lms = lms_at("CDC", "male", age * DAYS_PER_YEAR)

    def at(bmi):
        return assess(profile(age_years=age, sex="male", weight_kg=bmi * h * h, height_m=h))

    assert at(lms.bmi_at_percentile(50)).category == "Healthy weight"
    assert at(lms.bmi_at_percentile(90)).category == "Overweight"
    assert at(lms.bmi_at_percentile(96)).category == "Obesity"
    assert at(lms.bmi_at_percentile(95) * 1.25).category == "Severe obesity (class 2)"
    assert at(lms.bmi_at_percentile(2)).category == "Underweight"
    assert at(lms.bmi_at_percentile(50)).method == "CDC BMI-for-age percentile"


def test_adult_sample_and_weight_goal():
    a = assess(profile())
    assert (a.bmi, a.category, a.risk_level) == (25.2, "Overweight", "Moderate")
    assert (a.healthy_bmi_min, a.healthy_bmi_max) == (18.5, 24.9)
    assert a.weight_to_healthy_kg == -0.6 and a.bmi_prime == 1.01
    assert assess(profile(weight_kg=40, height_m=1.7)).weight_to_healthy_kg > 0
    assert assess(profile(weight_kg=65, height_m=1.7)).weight_to_healthy_kg == 0


def test_older_adult_gets_note_up_to_150():
    a = assess(profile(age_years=150, sex=None, weight_kg=60, height_m=1.6))
    assert a.life_stage == "older_adult" and a.notes


@pytest.mark.parametrize(
    "overrides",
    [
        {"height_m": 157},  # centimetres typed as metres
        {"age_years": -1},
        {"age_years": 151},
        {"weight_kg": 0},
        {"name": ""},
        {"age_years": 10, "sex": None},  # sex required under 20
        {"weight_kg": 300, "height_m": 1.0},  # implausible BMI
    ],
)
def test_profile_validation_rejects(overrides):
    with pytest.raises(ValidationError):
        profile(**overrides)


def test_adult_sex_is_optional():
    assert profile(sex=None).sex is None


def test_format_age():
    assert format_age(1 / DAYS_PER_YEAR) == "1 day"
    assert format_age(18 / 12) == "18 months"
    assert format_age(3.5) == "3 years 6 months"
    assert format_age(49) == "49 years"
