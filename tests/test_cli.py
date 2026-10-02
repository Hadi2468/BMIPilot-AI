import json

from bmipilot.cli import main, parse_args


def test_age_and_height_units():
    args = parse_args(["--age-days", "1", "--sex", "female", "--height-cm", "50"])
    assert args.age_years == 1 / 365.25 and args.height_m == 0.5
    assert parse_args(["--age-months", "18"]).age_years == 1.5
    default = parse_args([])
    assert (default.age_years, default.sex, default.height_m) == (49, "female", 1.57)


def test_bmi_only_needs_no_api_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = main(
        [
            "--age-days",
            "1",
            "--sex",
            "male",
            "--weight",
            "3.4",
            "--height-cm",
            "50",
            "--bmi-only",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["assessment"]["age_group"] == "infant_toddler"


def test_invalid_input_exits_2(capsys):
    assert main(["--height", "157", "--bmi-only"]) == 2
    assert main(["--age", "10", "--weight", "30", "--height", "1.3", "--bmi-only"]) == 2
    assert "sex is required" in capsys.readouterr().out
