import json
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from api.app.main import app
from api.app import config
from api.app import logger as logger_module
from api.app import db as db_module
from api.app.db import metadata


#  Container for temp references
class StorageContext:
    """Holds temp log path and sync engine for test assertions."""
    def __init__(self, log_path, sync_engine):
        self.log_path = log_path
        self.sync_engine = sync_engine

#  Test Isolation 
# Every test gets a fresh temp JSONL file AND a fresh temp SQLite DB.
# No test touches the real data/ directory or database.

@pytest.fixture(autouse=True)
def storage(tmp_path, monkeypatch):
    """Redirect JSONL and SQLite to temp locations for full hermiticity."""
    #  Temp JSONL 
    temp_log = tmp_path / "test_telemetry.jsonl"
    # Patch in both modules because Python imports are copies, not references
    monkeypatch.setattr(config, "LOG_PATH", temp_log)
    monkeypatch.setattr(logger_module, "LOG_PATH", temp_log)

    #  Temp SQLite 
    temp_db = tmp_path / "test.sqlite3"
    sync_url = f"sqlite:///{temp_db}"
    async_url = f"sqlite+aiosqlite:///{temp_db}"

    # Create a temp sync engine and initialize schema
    temp_sync_engine = create_engine(sync_url, echo=False)

    @event.listens_for(temp_sync_engine, "connect")
    def set_wal_sync(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    metadata.create_all(temp_sync_engine)

    # Create a temp async engine for the logger's SQLite mirror
    temp_async_engine = create_async_engine(async_url, echo=False)

    @event.listens_for(temp_async_engine.sync_engine, "connect")
    def set_wal_async(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    temp_async_session = sessionmaker(
        temp_async_engine, class_=AsyncSession, expire_on_commit=False
    )

    # Patch the db module so logger.py uses the temp DB
    monkeypatch.setattr(db_module, "async_engine", temp_async_engine)
    monkeypatch.setattr(db_module, "AsyncSessionLocal", temp_async_session)

    # Yield the container — tests access it via the 'storage' parameter
    yield StorageContext(log_path=temp_log, sync_engine=temp_sync_engine)


#  Test Client 
client = TestClient(app, raise_server_exceptions=False)


#  Helper 

def get_last_record(log_path) -> dict:
    """Read the last JSONL line and parse it as a dict."""
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    return json.loads(lines[-1])


#                JSONL Pipeline Tests 
class TestJSONLWriting:
    """Tests that the JSONL source of truth is written correctly."""
    def test_jsonl_written_on_request(self, storage):
        """Hitting any endpoint should append a line to the JSONL file."""
        client.post("/hp/login", json={"username": "test", "password": "test"})
        assert storage.log_path.exists()
        lines = storage.log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 1

    def test_jsonl_contains_headers(self, storage):
        """JSONL record must include a 'headers' dict."""
        client.post("/hp/balance", json={"account_id": "1234"})
        record = get_last_record(storage.log_path)
        assert "headers" in record
        assert isinstance(record["headers"], dict)
        # Should contain standard HTTP headers
        assert "user-agent" in record["headers"]
        assert "content-type" in record["headers"]

    def test_jsonl_contains_payload_error_on_bad_json(self, storage):
        """Sending garbage body should produce payload_error: 'invalid_json'."""
        client.post(
            "/hp/login",
            content=b"this is not json",
            headers={"content-type": "application/json"},
        )
        record = get_last_record(storage.log_path)
        assert record["payload"] is None
        assert record["payload_error"] == "invalid_json"

    def test_wrong_content_type_recorded(self, storage):
        """Sending text/plain should produce payload_error: 'wrong_content_type'."""
        client.post(
            "/hp/login",
            content=b"some plain text",
            headers={"content-type": "text/plain"},
        )
        record = get_last_record(storage.log_path)
        assert record["payload"] is None
        assert record["payload_error"] == "wrong_content_type"


#  SQLite Mirror Tests 


class TestSQLiteMirror:
    """Tests that the SQLite query mirror receives data."""
    def test_sqlite_mirrors_jsonl(self, storage):
        """After a request, the SQLite telemetry table should have a row."""
        client.post("/hp/checkout", json={"items": ["widget"]})
        with storage.sync_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM telemetry")).fetchone()[0]
        assert count >= 1

    def test_sqlite_failure_does_not_crash(self, storage, monkeypatch):
        """If SQLite is broken, the endpoint should still return normally
        and JSONL should still have the record."""
        # Break the async insert by making it always raise
        async def broken_insert(record):
            raise RuntimeError("Simulated DB failure")

        monkeypatch.setattr(db_module, "async_insert_telemetry", broken_insert)

        # The endpoint should still work
        response = client.post("/hp/login", json={"username": "x", "password": "y"})
        assert response.status_code == 401

        # JSONL should still have the record
        record = get_last_record(storage.log_path)
        assert record["endpoint"] == "/hp/login"


#        Session ID Tests
class TestSessionID:
    """Tests for session ID assignment logic."""
    def test_session_id_from_header(self, storage):
        """If X-Session-ID header is sent, JSONL should use that value."""
        custom_id = "my-custom-session-123"
        client.post(
            "/hp/login",
            json={"username": "test", "password": "test"},
            headers={"X-Session-ID": custom_id},
        )
        record = get_last_record(storage.log_path)
        assert record["session_id"] == custom_id

    def test_session_id_uuid_fallback(self, storage):
        """Without X-Session-ID, JSONL should have a valid UUID4."""
        client.post("/hp/login", json={"username": "test", "password": "test"})
        record = get_last_record(storage.log_path)
        # Should not raise, confirms it's a valid UUID
        parsed = uuid.UUID(record["session_id"], version=4)
        assert str(parsed) == record["session_id"]


#  Data Integrity Tests 

class TestDataIntegrity:
    """Tests for IP hashing and payload sanitization."""
    def test_ip_hashed_in_record(self, storage):
        """JSONL should contain ip_hash (hex string), not a raw IP address."""
        client.post("/hp/balance", json={"account_id": "1234"})
        record = get_last_record(storage.log_path)
        assert "ip_hash" in record
        # SHA-256 hex digest is always 64 characters
        assert len(record["ip_hash"]) == 64
        # Should NOT contain a raw IP like 127.0.0.1
        assert "127.0.0.1" not in record["ip_hash"]

    def test_payload_sanitized(self, storage):
        """Sending a password field should result in [REDACTED] in JSONL."""
        client.post(
            "/hp/login",
            json={"username": "admin", "password": "super_secret_123"},
        )
        record = get_last_record(storage.log_path)
        assert record["payload"]["password"] == "[REDACTED]"
        # Username should NOT be redacted
        assert record["payload"]["username"] == "admin"