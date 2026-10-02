"""Command-line entry point.

bmipilot --name Sara --sex female --age 35 --weight 70 --height 1.65
bmipilot --name Leo --sex male --age 9 --weight 38 --height-cm 135
bmipilot --name Baby --sex female --age-days 1 --weight 3.3 --height-cm 50 --bmi-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from bmipilot.assessment import assess
from bmipilot.config import get_settings
from bmipilot.growth import DAYS_PER_YEAR
from bmipilot.observability import log_tracing_status
from bmipilot.policy import DISCLAIMER
from bmipilot.report import format_tip, metric_rows
from bmipilot.schemas import Assessment, CoachResult, Profile

logger = logging.getLogger("bmipilot")

RISK_COLOURS = {"Low": "green", "Moderate": "yellow"}


def _print_assessment(profile: Profile, a: Assessment, console: Console) -> None:
    colour = RISK_COLOURS.get(a.risk_level, "red")
    table = Table(title=f"BMI Health Report: {profile.name}", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    for label, value in metric_rows(profile, a):
        table.add_row(label, f"[{colour}]{value}[/{colour}]" if label == "Risk level" else value)
    console.print(table)
    for note in a.notes:
        console.print(f"[cyan]Note:[/cyan] {note}")


def _print_result(result: CoachResult, console: Console) -> None:
    _print_assessment(result.profile, result.assessment, console)
    for title, body in [
        ("Exercise Plan", result.exercise_plan),
        ("Diet Tip", format_tip(result.diet_tip)),
        ("Lifestyle Tip", format_tip(result.lifestyle_tip)),
        ("Explanation", result.explanation),
    ]:
        console.print(f"\n[bold cyan]{title}[/bold cyan]")
        console.print(Markdown(body))
    for warning in result.warnings:
        console.print(f"\n[yellow]Warning:[/yellow] {warning}")
    console.print(f"\n[dim italic]{DISCLAIMER}[/dim italic]")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BMIPilot AI: age-aware BMI health coach (0-150 years)."
    )
    parser.add_argument("--name", default="Mrs. ST")
    parser.add_argument("--sex", choices=["female", "male"], help="Required under 20")
    age = parser.add_mutually_exclusive_group()
    age.add_argument("--age", type=float, help="Age in years (default 49)")
    age.add_argument("--age-months", type=float, help="Age in months")
    age.add_argument("--age-days", type=float, help="Age in days")
    parser.add_argument("--weight", type=float, default=62, help="Weight in kg")
    height = parser.add_mutually_exclusive_group()
    height.add_argument("--height", type=float, help="Height in metres (default 1.57)")
    height.add_argument("--height-cm", type=float, help="Height in centimetres")
    parser.add_argument("--bmi-only", action="store_true", help="Assessment only, no LLM")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON")
    parser.add_argument("--output", type=Path, help="Also write the result as JSON to a file")
    parser.add_argument("--save-graph", type=Path, help="Save the graph diagram as PNG and exit")
    args = parser.parse_args(argv)

    if args.age_days is not None:
        args.age_years = args.age_days / DAYS_PER_YEAR
    elif args.age_months is not None:
        args.age_years = args.age_months / 12
    elif args.age is not None:
        args.age_years = args.age
    else:  # default sample profile
        args.age_years = 49
        args.sex = args.sex or "female"
    args.height_m = args.height_cm / 100 if args.height_cm else (args.height or 1.57)
    return args


def _save_graph(path: Path) -> None:
    from bmipilot.graph import build_default_graph

    drawable = build_default_graph().get_graph()
    try:
        path.write_bytes(drawable.draw_mermaid_png())
        logger.info("Graph saved to %s", path)
    except Exception:
        # draw_mermaid_png calls a remote rendering API; fall back to Mermaid text.
        path.with_suffix(".mmd").write_text(drawable.draw_mermaid(), encoding="utf-8")
        logger.warning("PNG render failed; Mermaid source saved to %s", path.with_suffix(".mmd"))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    console = Console()

    if args.save_graph:
        _save_graph(args.save_graph)
        return 0

    try:
        profile = Profile(
            name=args.name,
            age_years=args.age_years,
            sex=args.sex,
            weight_kg=args.weight,
            height_m=args.height_m,
        )
    except ValidationError as exc:
        console.print("[red]Invalid input:[/red]")
        for err in exc.errors():
            field = err["loc"][0] if err["loc"] else "input"
            console.print(f"  - {field}: {err['msg'].removeprefix('Value error, ')}")
        return 2

    if args.bmi_only:
        assessment = assess(profile)
        payload = {"profile": profile.model_dump(), "assessment": assessment.model_dump()}
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            _print_assessment(profile, assessment, console)
        if args.output:
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0

    settings = get_settings()  # loads .env first, so API keys and tracing vars are visible
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY is missing. Add it to your .env file.[/red]")
        return 1
    log_tracing_status()

    from bmipilot.service import BMIPilotService

    with console.status("Running BMIPilot agent..."):
        result = BMIPilotService.from_settings(settings).coach(profile)

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _print_result(result, console)
    if args.output:
        args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
