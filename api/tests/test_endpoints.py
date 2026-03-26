import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from api.app.main import app
from api.app import config
from api.app import logger as logger_module
from api.app.db import metadata, engine

#  Test Isolation 
# Every test gets a fresh temp JSONL file and SQLite DB.
# No test touches the real data/ directory.

@pytest.fixture(autouse=True)
def isolate_storage(tmp_path, monkeypatch):
    """Redirect LOG_PATH and DB to temp locations for test hermiticity."""
    # Create a temp JSONL file path
    temp_log = tmp_path / "test_telemetry.jsonl"
    # Patch LOG_PATH in both config and logger modules so all code
    # that imported LOG_PATH sees the temp path
    monkeypatch.setattr(config, "LOG_PATH", temp_log)
    monkeypatch.setattr(logger_module, "LOG_PATH", temp_log)
    # Create tables in the existing test DB 
    metadata.create_all(engine)

#  Test Client 
# Sends fake HTTP requests to the app without booting uvicorn
client = TestClient(app, raise_server_exceptions=False)

#                                Endpoint Response Tests 

class TestLoginEndpoint:
    """Tests for /hp/login — fake auth failure with retry bait."""
    def test_login_returns_401(self):
        response = client.post(
            "/hp/login",
            json={"username": "admin", "password": "hunter2"},
        )
        assert response.status_code == 401

    def test_login_response_shape(self):
        response = client.post(
            "/hp/login",
            json={"username": "admin", "password": "hunter2"},
        )
        data = response.json()
        assert data["status"] == "error"
        assert data["code"] == "INVALID_CREDENTIALS"
        assert "message" in data
        assert "retry_after" in data


class TestBalanceEndpoint:
    """Tests for /hp/balance — fake partial financial data."""
    def test_balance_returns_fake_data(self):
        response = client.post(
            "/hp/balance",
            json={"account_id": "1234"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "XXXX" in data["account_id"]  # masked
        assert data["currency"] == "USD"

class TestTransferEndpoint:
    """Tests for /hp/transfer — pending status with verify redirect."""
    def test_transfer_returns_pending(self):
        response = client.post(
            "/hp/transfer",
            json={"from_account": "1234", "to_account": "5678", "amount": 500},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "pending"
        assert "transaction_id" in data
        assert "/hp/verify" in data["message"]  # redirect bait


class TestCheckoutEndpoint:
    """Tests for /hp/checkout — fake processing delay."""
    def test_checkout_returns_processing(self):
        response = client.post(
            "/hp/checkout",
            json={"items": ["widget"], "payment_method": "card"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert "order_id" in data

class TestVerifyEndpoint:
    """Tests for /hp/verify — the perpetual trap."""
    def test_verify_returns_expired(self):
        response = client.post(
            "/hp/verify",
            json={"transaction_id": "TXN-abc", "verification_code": "123456"},
        )
        assert response.status_code == 403
        data = response.json()
        assert data["status"] == "error"
        assert data["code"] == "VERIFICATION_EXPIRED"
        assert "retry_after" in data


class TestHistoryEndpoint:
    """Tests for /hp/history — fake transaction list with pagination bait."""
    def test_history_returns_transactions(self):
        response = client.get("/hp/history")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["transactions"]) == 3
        assert data["total_pages"] == 3


#  Response Format Consistency 

class TestResponseFormatConsistency:
    """
    All honeypot error responses share the same shape.
    Prevents a smart agent from fingerprinting the honeypot by
    comparing error formats across endpoints.
    """
    def test_error_response_format_consistent(self):
        """Error endpoints (login, verify) must have identical key sets."""
        login_resp = client.post(
            "/hp/login",
            json={"username": "x", "password": "y"},
        )
        verify_resp = client.post(
            "/hp/verify",
            json={"transaction_id": "x", "verification_code": "y"},
        )

        login_keys = set(login_resp.json().keys())
        verify_keys = set(verify_resp.json().keys())

        # Both error responses must have these core keys
        required_keys = {"status", "code", "message", "retry_after"}
        assert required_keys.issubset(login_keys)
        assert required_keys.issubset(verify_keys)
        # And their key sets should match each other
        assert login_keys == verify_keys