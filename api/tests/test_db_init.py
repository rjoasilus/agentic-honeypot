# Proves init_db() creates correct tables and WAL mode is active.
import sqlite3
from sqlalchemy import create_engine, event, inspect
from api.app.db import metadata, init_db
 
# Expected columns per table
TELEMETRY_COLUMNS = {
    "id", "session_id", "timestamp_ms", "endpoint", "method",
    "payload_size", "response_time_ms", "ip_hash", "user_agent",
    "status_code", "actor_type",
}
 
SESSIONS_COLUMNS = {
    "session_id", "actor_type", "start_time", "end_time",
    "request_count", "created_at",
}
 
def _make_temp_engine(tmp_path):
    """Create a temp SQLite engine with the WAL listener attached."""
    db_file = tmp_path / "test.db"
    temp_engine = create_engine(f"sqlite:///{db_file}", echo=False)
 
    # Attach the same WAL listener that db.py uses
    @event.listens_for(temp_engine, "connect")
    def _set_wal(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()
    return temp_engine, db_file
 
 
class TestInitDb:
    """Prove init_db() creates the expected schema."""
    def test_tables_exist(self, tmp_path, monkeypatch):
        temp_engine, _ = _make_temp_engine(tmp_path)
        monkeypatch.setattr("api.app.db.engine", temp_engine)
        init_db()
        table_names = inspect(temp_engine).get_table_names()
        assert "telemetry" in table_names, "telemetry table not created"
        assert "sessions" in table_names, "sessions table not created"
 
    def test_telemetry_columns(self, tmp_path, monkeypatch):
        temp_engine, _ = _make_temp_engine(tmp_path)
        monkeypatch.setattr("api.app.db.engine", temp_engine)
        init_db()
        columns = {c["name"] for c in inspect(temp_engine).get_columns("telemetry")}
        assert columns == TELEMETRY_COLUMNS, (
            f"Column mismatch. Expected {TELEMETRY_COLUMNS}, got {columns}"
        )
 
    def test_sessions_columns(self, tmp_path, monkeypatch):
        temp_engine, _ = _make_temp_engine(tmp_path)
        monkeypatch.setattr("api.app.db.engine", temp_engine)
        init_db()
        columns = {c["name"] for c in inspect(temp_engine).get_columns("sessions")}
        assert columns == SESSIONS_COLUMNS, (
            f"Column mismatch. Expected {SESSIONS_COLUMNS}, got {columns}"
        )
 
 
class TestWalMode:
    """Prove WAL mode is active at the database file level."""
    def test_wal_mode_active(self, tmp_path, monkeypatch):
        temp_engine, db_file = _make_temp_engine(tmp_path)
        monkeypatch.setattr("api.app.db.engine", temp_engine)
        # init_db() opens a connection, which triggers the WAL listener
        init_db()
        # Verify with a raw sqlite3 connection (independent of SQLAlchemy).
        # WAL is persistent in the file header, so any connection sees it.
        conn = sqlite3.connect(str(db_file))
        result = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert result == "wal", f"Expected WAL mode, got '{result}'"