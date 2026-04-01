"""Tests for the CropRL REST API (extras/api)."""

import pytest
from fastapi.testclient import TestClient

from extras.api.app import app


@pytest.fixture(autouse=True)
def _reset_global_state():
    """Reset global env state between tests."""
    import extras.api.app as api_mod
    api_mod._env = None
    api_mod._last_obs = None
    api_mod._episode_step = 0
    yield


@pytest.fixture
def client():
    return TestClient(app)


# ── Health & Info ──────────────────────────────────────────────


class TestInfo:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_list_tasks(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200
        data = r.json()
        task_ids = [t["task_id"] for t in data]
        assert "easy" in task_ids
        assert "medium" in task_ids
        assert "hard" in task_ids

    def test_list_actions(self, client):
        r = client.get("/actions")
        assert r.status_code == 200
        actions = r.json()["actions"]
        assert len(actions) == 11
        assert actions[0]["id"] == 0
        assert actions[0]["name"] == "Wait"


# ── Reset ──────────────────────────────────────────────────────


class TestReset:
    def test_reset_default(self, client):
        r = client.post("/reset")
        assert r.status_code == 200
        data = r.json()
        assert data["task_id"] == "medium"
        assert "observation" in data
        obs = data["observation"]
        assert obs["current_month"] == 1
        assert obs["cash_balance"] == 10000.0
        assert obs["done"] is False

    def test_reset_easy(self, client):
        r = client.post("/reset", json={"task": "easy", "seed": 1})
        assert r.status_code == 200
        obs = r.json()["observation"]
        assert obs["cash_balance"] == 15000.0

    def test_reset_hard(self, client):
        r = client.post("/reset", json={"task": "hard"})
        assert r.status_code == 200
        obs = r.json()["observation"]
        assert obs["cash_balance"] == 7000.0

    def test_reset_unknown_task(self, client):
        r = client.post("/reset", json={"task": "impossible"})
        assert r.status_code == 400

    def test_reset_with_text_mode(self, client):
        r = client.post("/reset", json={"task": "easy", "text_mode": True})
        obs = r.json()["observation"]
        assert obs["text_summary"] is not None
        assert "Farm Dashboard" in obs["text_summary"]

    def test_reset_without_text_mode(self, client):
        r = client.post("/reset", json={"task": "easy", "text_mode": False})
        obs = r.json()["observation"]
        assert obs["text_summary"] is None


# ── State ──────────────────────────────────────────────────────


class TestState:
    def test_state_before_reset(self, client):
        r = client.get("/state")
        assert r.status_code == 400

    def test_state_after_reset(self, client):
        client.post("/reset")
        r = client.get("/state")
        assert r.status_code == 200
        data = r.json()
        assert data["step"] == 0
        assert data["done"] is False
        assert "observation" in data


# ── Step ───────────────────────────────────────────────────────


class TestStep:
    def test_step_before_reset(self, client):
        r = client.post("/step", json={"action_id": 0})
        assert r.status_code == 400

    def test_step_wait(self, client):
        client.post("/reset")
        r = client.post("/step", json={"action_id": 0})
        assert r.status_code == 200
        data = r.json()
        assert data["step"] == 1
        assert data["action_id"] == 0
        assert data["action_name"] == "Wait"
        assert "reward" in data
        assert data["done"] is False

    def test_step_plant_chickpea(self, client):
        client.post("/reset")
        r = client.post("/step", json={"action_id": 3})
        assert r.status_code == 200
        data = r.json()
        assert data["action_name"] == "Plant Chickpea (Legume)"
        assert data["observation"]["active_crop_type"] == 3
        assert "planted" in data["message"].lower()

    def test_step_invalid_action_id(self, client):
        client.post("/reset")
        r = client.post("/step", json={"action_id": 99})
        assert r.status_code in (400, 422)

    def test_step_after_done(self, client):
        """Stepping after episode is done should return 400."""
        client.post("/reset", json={"task": "easy"})
        # Run until done
        for _ in range(70):
            r = client.post("/step", json={"action_id": 0})
            if r.json()["done"]:
                break
        # Try one more step
        r = client.post("/step", json={"action_id": 0})
        assert r.status_code == 400

    def test_observation_structure(self, client):
        """Verify the observation dict has all expected keys."""
        client.post("/reset")
        r = client.post("/step", json={"action_id": 0})
        obs = r.json()["observation"]
        required_keys = [
            "current_month", "current_step", "expected_rainfall",
            "active_crop_type", "crop_age_months", "soil_nitrogen",
            "cash_balance", "current_debt", "market_prices",
            "costs", "storage", "done",
        ]
        for key in required_keys:
            assert key in obs, f"Missing key: {key}"


# ── Full Episode ───────────────────────────────────────────────


class TestFullEpisode:
    def test_plant_harvest_cycle(self, client):
        """Run a full plant → wait → harvest cycle via API."""
        client.post("/reset", json={"task": "easy"})

        # Plant chickpea
        r = client.post("/step", json={"action_id": 3})
        assert r.json()["observation"]["active_crop_type"] == 3

        # Wait
        r = client.post("/step", json={"action_id": 0})
        assert r.json()["observation"]["crop_age_months"] >= 1

        # Harvest & sell
        r = client.post("/step", json={"action_id": 7})
        data = r.json()
        assert data["observation"]["active_crop_type"] == 0
        assert data["reward"] > 0

    def test_state_updates_after_step(self, client):
        """GET /state should reflect changes from POST /step."""
        client.post("/reset")

        # Take action
        client.post("/step", json={"action_id": 3})

        # Verify state matches
        r = client.get("/state")
        data = r.json()
        assert data["step"] == 1
        assert data["observation"]["active_crop_type"] == 3

    def test_multiple_resets(self, client):
        """Resetting should fully clear previous episode."""
        client.post("/reset", json={"task": "easy"})
        client.post("/step", json={"action_id": 3})
        client.post("/step", json={"action_id": 0})

        # Reset again
        r = client.post("/reset", json={"task": "hard"})
        obs = r.json()["observation"]
        assert obs["cash_balance"] == 7000.0
        assert obs["active_crop_type"] == 0
        assert obs["current_step"] == 0
