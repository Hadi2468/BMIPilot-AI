"""BMI-for-age reference curves for children, using the LMS method.

Data: `data/bmi_for_age_lms.csv` (ages in days)
- WHO Child Growth Standards, BMI-for-age, 0-731 days (daily rows from the WHO
  expanded z-score tables, boys and girls).
- CDC 2000 Growth Charts, BMI-for-age, 24-240.5 months (`bmiagerev.csv`).

CDC recommends the WHO standards under 2 years and the CDC charts from 2 to 19 years.
"""

from __future__ import annotations

import csv
import math
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from statistics import NormalDist
from typing import Literal

Sex = Literal["female", "male"]
Source = Literal["WHO", "CDC"]

DAYS_PER_YEAR = 365.25
WHO_MAX_DAYS = 730.5  # 24 months: WHO below, CDC from here

_NORMAL = NormalDist()


@dataclass(frozen=True)
class LMS:
    """Box-Cox power (L), median (M) and coefficient of variation (S) at one age."""

    L: float
    M: float
    S: float

    def z_score(self, bmi: float) -> float:
        if abs(self.L) < 1e-9:
            return math.log(bmi / self.M) / self.S
        return ((bmi / self.M) ** self.L - 1) / (self.L * self.S)

    def bmi_at_z(self, z: float) -> float:
        if abs(self.L) < 1e-9:
            return self.M * math.exp(self.S * z)
        return self.M * (1 + self.L * self.S * z) ** (1 / self.L)

    def bmi_at_percentile(self, percentile: float) -> float:
        return self.bmi_at_z(_NORMAL.inv_cdf(percentile / 100))


@lru_cache
def _table(source: Source, sex: Sex) -> tuple[tuple[float, ...], tuple[LMS, ...]]:
    path = resources.files("bmipilot") / "data" / "bmi_for_age_lms.csv"
    with path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] == source and r["sex"] == sex]
    ages = tuple(float(r["age_days"]) for r in rows)
    lms = tuple(LMS(float(r["L"]), float(r["M"]), float(r["S"])) for r in rows)
    return ages, lms


def lms_at(source: Source, sex: Sex, age_days: float) -> LMS:
    """LMS parameters at an exact age, linearly interpolated between table rows."""
    ages, lms = _table(source, sex)
    age_days = min(max(age_days, ages[0]), ages[-1])  # clamp to the table's range
    i = bisect_left(ages, age_days)
    if ages[i] == age_days:
        return lms[i]
    lo, hi = lms[i - 1], lms[i]
    t = (age_days - ages[i - 1]) / (ages[i] - ages[i - 1])
    return LMS(lo.L + t * (hi.L - lo.L), lo.M + t * (hi.M - lo.M), lo.S + t * (hi.S - lo.S))


def who_z_score(lms: LMS, bmi: float) -> float:
    """WHO z-score, with WHO's restricted rule beyond +/-3 SD (keeps the tails linear)."""
    z = lms.z_score(bmi)
    if z > 3:
        sd3, sd2 = lms.bmi_at_z(3), lms.bmi_at_z(2)
        return 3 + (bmi - sd3) / (sd3 - sd2)
    if z < -3:
        sd3, sd2 = lms.bmi_at_z(-3), lms.bmi_at_z(-2)
        return -3 + (bmi - sd3) / (sd2 - sd3)
    return z


def percentile_from_z(z: float) -> float:
    return _NORMAL.cdf(z) * 100


def reference_curves(source: Source, sex: Sex, step_days: float = 15) -> list[dict[str, float]]:
    """Curves for charts: WHO as z-score lines (-2, 0, +1, +2), CDC as percentiles (5-95)."""
    ages, _ = _table(source, sex)
    start = ages[0] if source == "WHO" else WHO_MAX_DAYS
    end = WHO_MAX_DAYS if source == "WHO" else 20 * DAYS_PER_YEAR
    points = []
    age = start
    while age <= end:
        lms = lms_at(source, sex, age)
        point = {"age_years": round(age / DAYS_PER_YEAR, 3)}
        if source == "WHO":
            point |= {f"z{z:+d}": round(lms.bmi_at_z(z), 2) for z in (-2, 0, 1, 2)}
        else:
            point |= {f"p{p}": round(lms.bmi_at_percentile(p), 2) for p in (5, 50, 85, 95)}
        points.append(point)
        age += step_days
    return points
