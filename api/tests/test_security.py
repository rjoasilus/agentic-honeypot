from fastapi.testclient import TestClient
from api.app.main import app
from api.app.sanitizer import sanitize_payload

# TestClient simulates HTTP requests without booting a real server
client = TestClient(app)

def test_payload_sanitizer_redacts_password():
    """Verify sensitive top-level fields get replaced with [REDACTED]."""
    payload = {"user": "test", "password": "supersecret", "token": "abc123"}
    sanitized = sanitize_payload(payload)
    assert sanitized["user"] == "test"
    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["token"] == "[REDACTED]"

def test_payload_sanitizer_handles_nested():
    """Verify sensitive fields inside nested dicts also get redacted."""
    payload = {"data": {"credit_card": "4111111111111111", "name": "Project"}}
    sanitized = sanitize_payload(payload)
    assert sanitized["data"]["credit_card"] == "[REDACTED]"
    assert sanitized["data"]["name"] == "Project"

def test_oversized_payload_rejected():
    """Payloads over MAX_PAYLOAD_BYTES should return 413."""
    oversized = {"data": "x" * 20000}
    response = client.post(
        "/hp/login",
        json=oversized,
    )
    assert response.status_code == 413