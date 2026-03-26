"""Tests for features/etl.py — JSONL loading, schema validation, sorting."""
import json
import pandas as pd
import pytest
from features.etl import (
    REQUIRED_COLUMNS,
    VALID_ACTOR_TYPES,
    load_jsonl,
    sort_by_session_and_time,
    validate_schema,
)

#  Fixtures 

def _make_record(**overrides):
    """Build a valid JSONL record with sensible defaults. Override any field."""
    base = {
        "session_id": "test-session-001",
        "timestamp_ms": 1700000000000,
        "endpoint": "/hp/login",
        "method": "POST",
        "payload_size": 42,
        "response_time_ms": 1.5,
        "ip_hash": "abc123",
        "user_agent": "Mozilla/5.0",
        "headers": {"content-type": "application/json"},
        "status_code": 401,
        "actor_type": "human",
        "payload": {"username": "test"},
        "payload_error": None,
    }
    base.update(overrides)
    return base


def _write_jsonl(path, records):
    """Write a list of dicts as JSONL to the given path."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")

#  load_jsonl 

class TestLoadJsonl:
    def test_loads_valid_records(self, tmp_path):
        """Valid JSONL -> correct row count and columns."""
        path = tmp_path / "test.jsonl"
        records = [_make_record(timestamp_ms=1000 + i) for i in range(5)]
        _write_jsonl(path, records)

        df = load_jsonl(path)
        assert len(df) == 5
        for col in REQUIRED_COLUMNS:
            assert col in df.columns

    def test_skips_malformed_lines(self, tmp_path):
        """One bad line doesn't kill the load. skips it, keeps the rest."""
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(_make_record(timestamp_ms=1000)) + "\n")
            f.write("THIS IS NOT JSON\n")  # malformed
            f.write(json.dumps(_make_record(timestamp_ms=2000)) + "\n")

        df = load_jsonl(path)
        assert len(df) == 2  # skipped the bad line

    def test_empty_file_returns_empty_df(self, tmp_path):
        """Empty JSONL -> empty DataFrame with correct columns."""
        path = tmp_path / "empty.jsonl"
        path.write_text("")

        df = load_jsonl(path)
        assert df.empty
        for col in REQUIRED_COLUMNS:
            assert col in df.columns

    def test_missing_file_returns_empty_df(self, tmp_path):
        """Non-existent file → empty DataFrame, no crash."""
        path = tmp_path / "does_not_exist.jsonl"
        df = load_jsonl(path)
        assert df.empty


#  validate_schema 

class TestValidateSchema:
    def test_drops_null_session_id(self, tmp_path):
        """Rows with null session_id are removed."""
        records = [
            _make_record(session_id="valid-session", timestamp_ms=1000),
            _make_record(session_id=None, timestamp_ms=2000),
            _make_record(session_id="", timestamp_ms=3000),
        ]
        df = pd.DataFrame(records)
        result = validate_schema(df)
        assert len(result) == 1
        assert result.iloc[0]["session_id"] == "valid-session"

    def test_drops_invalid_actor_type(self, tmp_path):
        """Only human/bot/llm_agent survive, 'unknown' and others are dropped."""
        records = [
            _make_record(actor_type="human", timestamp_ms=1000),
            _make_record(actor_type="unknown", timestamp_ms=2000),
            _make_record(actor_type="bot", timestamp_ms=3000),
            _make_record(actor_type="alien", timestamp_ms=4000),
        ]
        df = pd.DataFrame(records)
        result = validate_schema(df)
        assert len(result) == 2
        assert set(result["actor_type"]) == {"human", "bot"}

    def test_drops_invalid_timestamp(self, tmp_path):
        """Rows with null or non-positive timestamp are removed."""
        records = [
            _make_record(timestamp_ms=1000),
            _make_record(timestamp_ms=0),
            _make_record(timestamp_ms=-100),
        ]
        df = pd.DataFrame(records)
        result = validate_schema(df)
        assert len(result) == 1

    def test_enforces_dtypes(self):
        """Numeric columns are cast to correct types after validation."""
        records = [_make_record()]
        df = pd.DataFrame(records)
        result = validate_schema(df)
        assert result["timestamp_ms"].dtype == "int64"
        assert result["payload_size"].dtype == "int64"
        assert result["status_code"].dtype == "int64"
        assert result["response_time_ms"].dtype == "float64"

    def test_raises_on_missing_columns(self):
        """Missing required columns -> ValueError, not silent failure."""
        df = pd.DataFrame({"session_id": ["abc"], "timestamp_ms": [1000]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_schema(df)

    def test_empty_df_passthrough(self):
        """Empty DataFrame passes through without error."""
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        result = validate_schema(df)
        assert result.empty


#  sort_by_session_and_time 

class TestSortBySessionAndTime:
    def test_sorts_within_session(self):
        """Rows within a session are sorted by timestamp ascending."""
        records = [
            _make_record(session_id="s1", timestamp_ms=300),
            _make_record(session_id="s1", timestamp_ms=100),
            _make_record(session_id="s1", timestamp_ms=200),
        ]
        df = validate_schema(pd.DataFrame(records))
        result = sort_by_session_and_time(df)
        assert list(result["timestamp_ms"]) == [100, 200, 300]

    def test_groups_sessions_together(self):
        """Rows from the same session are contiguous after sort."""
        records = [
            _make_record(session_id="s2", timestamp_ms=200),
            _make_record(session_id="s1", timestamp_ms=100),
            _make_record(session_id="s2", timestamp_ms=100),
            _make_record(session_id="s1", timestamp_ms=200),
        ]
        df = validate_schema(pd.DataFrame(records))
        result = sort_by_session_and_time(df)
        session_ids = list(result["session_id"])
        # s1 rows should be contiguous, s2 rows should be contiguous
        assert session_ids == ["s1", "s1", "s2", "s2"]