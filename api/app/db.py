from sqlalchemy import (
    create_engine, Column, Integer, Text, Float,
    Index, MetaData, Table, event
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from api.app.config import DB_PATH

#  Paths 
# Ensure the database directory exists before creating engines
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DB_PATH}"
ASYNC_DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

#  Engines 
# Sync engine: used only for init_db() at startup (table creation)
engine = create_engine(DATABASE_URL, echo=False)
# Async engine: used at runtime for non-blocking telemetry inserts
async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)
# Session factory: stamps out async sessions bound to the async engine
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

metadata = MetaData()

#            WAL Mode 
# Enable Write-Ahead Logging for concurrent read/write safety.
# set on BOTH engines since they maintain separate connection pools.

@event.listens_for(engine, "connect")
def set_sqlite_wal(dbapi_conn, connection_record):
    """Enable WAL mode on every sync engine connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()

@event.listens_for(async_engine.sync_engine, "connect")
def set_async_sqlite_wal(dbapi_conn, connection_record):
    """Enable WAL mode on every async engine connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()

#  Schema 
# These tables are created once at startup by init_db().
telemetry = Table(
    "telemetry",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", Text, nullable=False),
    Column("timestamp_ms", Integer, nullable=False),
    Column("endpoint", Text, nullable=False),
    Column("method", Text, nullable=False),
    Column("payload_size", Integer, default=0),
    Column("response_time_ms", Float, default=0.0),
    Column("ip_hash", Text),
    Column("user_agent", Text),
    Column("status_code", Integer),
    Column("actor_type", Text),  # human / bot / llm_agent / unknown
)

sessions = Table(
    "sessions",
    metadata,
    Column("session_id", Text, primary_key=True),
    Column("actor_type", Text, nullable=False),
    Column("start_time", Integer),
    Column("end_time", Integer),
    Column("request_count", Integer, default=0),
    Column("created_at", Text),
)

#  Indexes 
# Speed up GROUP BY session_id (feature engineering), ORDER BY timestamp_ms
# (gap computation), and WHERE actor_type = '...' (EDA filtering).
# Created by metadata.create_all() alongside tables — no-op if they already exist.
Index("ix_telemetry_session_id", telemetry.c.session_id)
Index("ix_telemetry_timestamp_ms", telemetry.c.timestamp_ms)
Index("ix_telemetry_actor_type", telemetry.c.actor_type)

#  Startup 
def init_db():
    """Create all tables if they don't exist. Called once at server boot."""
    metadata.create_all(engine)
    print(f"[DB] Initialized at {DB_PATH}")

# Runtime Insert 
async def async_insert_telemetry(record: dict) -> None:
    """Insert one telemetry row into SQLite (async, non-blocking).
    Only mirrors columns defined in the schema — headers, payload,
    and payload_error live exclusively in JSONL.
    Caller (logger.py) is responsible for try/except."""
    async with AsyncSessionLocal() as session:
        async with session.begin():
            await session.execute(
                telemetry.insert().values(
                    session_id=record["session_id"],
                    timestamp_ms=record["timestamp_ms"],
                    endpoint=record["endpoint"],
                    method=record["method"],
                    payload_size=record["payload_size"],
                    response_time_ms=record["response_time_ms"],
                    ip_hash=record["ip_hash"],
                    user_agent=record["user_agent"],
                    status_code=record["status_code"],
                    actor_type=record["actor_type"],
                )
            )

if __name__ == "__main__":
    init_db()