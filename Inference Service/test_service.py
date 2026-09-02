# test_service.py
import pytest
from server import app   # assuming your code is in server.py


@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


def test_health_root(client):
    """Health check at root endpoint should return 200"""
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Recommendation Service" in resp.data or b"SVD Recommendation Service" in resp.data


def test_health_check(client):
    """Detailed health endpoint should return 200 and model status"""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.data.decode("utf-8")
    assert "Service: Running" in body
    assert "SVD" in body
    assert "ALS" in body


def test_recommend_valid_user(client):
    """Valid user id should return 200 (may be empty or filled list)"""
    resp = client.get("/recommend/1")
    assert resp.status_code == 200
    # response may be empty or comma-separated recommendations
    assert isinstance(resp.data.decode("utf-8"), str)


def test_recommend_invalid_userid(client):
    """Non-integer user id should return 400"""
    resp = client.get("/recommend/abc")
    assert resp.status_code == 400


def test_recommend_wrong_method(client):
    """POST to recommend endpoint should return 405"""
    resp = client.post("/recommend/1")
    assert resp.status_code == 405


def test_non_existent_endpoint(client):
    """Unknown path should return 404"""
    resp = client.get("/doesnotexist")
    assert resp.status_code == 404
