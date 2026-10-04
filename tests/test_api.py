"""Tests for the F1 prediction API - each test gaurds a bug we actually hit."""

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


FAVORITE = {"grid": 1, "driver_form": 1.4, "constructor_strength": 28.5,
            "constructor": "Red Bull", "circuit": "Suzuka Circuit"}
BACKMARKER = {"grid": 18, "driver_form": 16.0, "constructor_strength": 0.5,
              "constructor": "Williams", "circuit": "Suzuka Circuit"}


def test_health(client):
    """Health endpoint should return 200 and model loaded."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_favorite_beats_backmarker(client):
    fav = client.post("/predict", json=FAVORITE).json()["win_probability"]
    back = client.post("/predict", json=BACKMARKER).json()["win_probability"]
    assert fav > back, f"Favorite {fav} should beat backmarker {back}"

def test_probability_in_unit_interval(client):
    p = client.post("/predict", json=FAVORITE).json()["win_probability"]
    assert 0.0 <= p <= 1.0, f"Probability {p} should be in [0,1]"


def test_invalid_grid_rejected(client):
    """Grid must be in [1,24]"""
    for bad in [0, 25, -1]:
        r = client.post("/predict", json={**FAVORITE, "grid": bad})
        assert r.status_code == 422, f"Bad grid {bad} should be rejected"

def test_unkown_constructor_is_graceful(client):
    """Unknown constructor should be treated as 0 in the one-hot vector."""
    r = client.post("/predict", json={**FAVORITE, "constructor": "NotAConstructor"})
    assert r.status_code == 200, "Unknown constructor should be accepted"
    p = r.json()["win_probability"]
    assert 0.0 <= p <= 1.0, f"Probability {p} should be in [0,1]"

def test_saved_model_artifact_is_filtered():
    """Regression test: the shipped artifact must be fitted"""

    bundle = joblib.load("model/model.joblib")
    model, cols = bundle["model"], bundle["feature_columns"]
    assert hasattr(model, "classes_"), "Saved model must be fitted"
    assert model.predict_proba(np.zeros((1,len(cols)))).shape == (1,2)