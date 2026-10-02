"""The LMS implementation must reproduce the values published by WHO and CDC."""

import pytest

from bmipilot.growth import lms_at, percentile_from_z, reference_curves, who_z_score

MONTH = 30.4375

# CDC bmiagerev.csv: (sex, age in months, P5, P50, P85, P95)
CDC_PUBLISHED = [
    ("male", 24, 14.73731947, 16.57502768, 18.16219473, 19.33801062),
    ("female", 30.5, 14.17991775, 16.00593001, 17.50964839, 18.5759903),
    ("female", 120.5, 14.03535405, 16.86231366, 19.98399779, 22.98257733),
    ("male", 216.5, 18.24348662, 21.8958685, 25.65601244, 28.9586172),
]

# WHO expanded BMI-for-age tables: (sex, age in days, -2 SD, median, +1 SD, +2 SD)
WHO_PUBLISHED = [
    ("male", 1, 11.095, 13.398, 14.76, 16.29),
    ("female", 365, 13.794, 16.358, 17.888, 19.62),
    ("male", 700, 13.607, 15.785, 17.097, 18.597),
]


@pytest.mark.parametrize("sex,months,p5,p50,p85,p95", CDC_PUBLISHED)
def test_cdc_percentiles_match_published(sex, months, p5, p50, p85, p95):
    lms = lms_at("CDC", sex, months * MONTH)
    for pct, expected in ((5, p5), (50, p50), (85, p85), (95, p95)):
        assert lms.bmi_at_percentile(pct) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("sex,days,sd2neg,sd0,sd1,sd2", WHO_PUBLISHED)
def test_who_sd_curves_match_published(sex, days, sd2neg, sd0, sd1, sd2):
    lms = lms_at("WHO", sex, days)
    for z, expected in ((-2, sd2neg), (0, sd0), (1, sd1), (2, sd2)):
        assert lms.bmi_at_z(z) == pytest.approx(expected, abs=0.01)


def test_z_score_round_trip():
    lms = lms_at("CDC", "female", 3000)
    assert lms.z_score(lms.bmi_at_z(1.3)) == pytest.approx(1.3)
    assert percentile_from_z(0) == pytest.approx(50)


def test_interpolates_between_rows():
    a, b = lms_at("CDC", "male", 730.5), lms_at("CDC", "male", 745.7188)
    mid = lms_at("CDC", "male", (730.5 + 745.7188) / 2)
    median = mid.M
    assert median == pytest.approx((a.M + b.M) / 2)


def test_who_restricted_rule_beyond_3sd():
    lms = lms_at("WHO", "male", 100)
    sd3, sd2 = lms.bmi_at_z(3), lms.bmi_at_z(2)
    assert who_z_score(lms, sd3 + (sd3 - sd2)) == pytest.approx(4)
    sd3n, sd2n = lms.bmi_at_z(-3), lms.bmi_at_z(-2)
    assert who_z_score(lms, sd3n - (sd2n - sd3n)) == pytest.approx(-4)


def test_reference_curves_cover_their_age_ranges():
    who = reference_curves("WHO", "female")
    cdc = reference_curves("CDC", "male")
    assert who[0]["age_years"] == 0 and who[-1]["age_years"] <= 2
    assert set(who[0]) == {"age_years", "z-2", "z+0", "z+1", "z+2"}
    assert cdc[0]["age_years"] == pytest.approx(2, abs=0.01) and cdc[-1]["age_years"] <= 20
    assert all(p["p5"] < p["p50"] < p["p85"] < p["p95"] for p in cdc)
