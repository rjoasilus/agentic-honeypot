from fastapi.testclient import TestClient
from api.app.main import app

# TestClient simulates HTTP requests without booting a real server
client = TestClient(app)

def test_health_returns_ok():
    """Verify the health endpoint confirms the server is alive."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "time" in data
