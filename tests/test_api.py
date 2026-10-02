import pytest
from fastapi.testclient import TestClient

from bmipilot.api import create_app
from bmipilot.service import BMIPilotService

ADULT = {"name": "Sara", "age_years": 35, "sex": "female", "weight_kg": 70, "height_m": 1.65}
BABY = {"name": "Baby", "age_years": 0.5, "sex": "male", "weight_kg": 7.9, "height_m": 0.68}


@pytest.fixture
def client_for(make_graph, settings):
    def _make(api_token=None):
        s = settings.model_copy(update={"api_token": api_token})
        return TestClient(create_app(service=BMIPilotService(make_graph(), s), settings=s))

    return _make


def test_health(client_for):
    with client_for() as client:
        assert client.get("/health").json()["status"] == "ok"


def test_assess_is_deterministic(client_for):
    with client_for() as client:
        adult = client.post("/assess", json=ADULT).json()
        baby = client.post("/assess", json=BABY).json()
    assert adult["age_group"] == "adult" and adult["category"] == "Overweight"
    assert baby["age_group"] == "infant_toddler" and baby["z_score"] is not None


def test_coach_runs_the_agent(client_for):
    with client_for() as client:
        body = client.post("/coach", json=ADULT).json()
    assert body["assessment"]["bmi"] == 25.7
    assert body["report"].startswith("# Health Report for Sara")


@pytest.mark.parametrize(
    "payload",
    [
        ADULT | {"height_m": 165},
        ADULT | {"age_years": 12, "sex": None},
        ADULT | {"age_years": 200},
    ],
)
def test_invalid_profile_returns_422(client_for, payload):
    with client_for() as client:
        assert client.post("/assess", json=payload).status_code == 422


def test_reference_curves(client_for):
    with client_for() as client:
        who = client.get("/reference/WHO/female").json()
        cdc = client.get("/reference/CDC/male").json()
        assert client.get("/reference/XYZ/male").status_code == 422
    assert "z+0" in who[0] and "p95" in cdc[0]


def test_bearer_token_is_enforced_when_configured(client_for):
    with client_for(api_token="s3cret") as client:
        assert client.post("/assess", json=ADULT).status_code == 401
        assert (
            client.post(
                "/assess", json=ADULT, headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        ok = client.post("/assess", json=ADULT, headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        # /health stays open for container health checks
        assert client.get("/health").status_code == 200
