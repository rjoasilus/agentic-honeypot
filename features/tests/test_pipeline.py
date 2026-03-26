"""Tests for features/pipeline.py — end-to-end pipeline integration tests."""
import json
from pathlib import Path
import pandas as pd
import pytest
from features.pipeline import run_pipeline

# Expected columns in the output CSV
EXPECTED_FEATURE_COLUMNS = [
    "session_id",
    "actor_type",
    "request_count",
    "mean_gap_time",
    "std_gap_time",
    "max_delay",
    "session_duration",
    "request_rate",
    "burstiness",
    "recovery_time",
    "session_entropy",
    "payload_entropy",
    "path_repetition",
]


def _make_record(session_id, actor_type, timestamp_ms, **overrides):
    """Build a valid JSONL record with required fields."""
    base = {
        "session_id": session_id,
        "timestamp_ms": timestamp_ms,
        "endpoint": "/hp/login",
        "method": "POST",
        "payload_size": 42,
        "response_time_ms": 1.5,
        "ip_hash": "abc123",
        "user_agent": "Mozilla/5.0",
        "headers": {"content-type": "application/json"},
        "status_code": 200,
        "actor_type": actor_type,
        "payload": {"username": "test"},
        "payload_error": None,
    }
    base.update(overrides)
    return base


def _write_test_jsonl(path):
    """small multi-session JSONL file for pipeline testing.
    3 sessions: 1 human (3 reqs), 1 bot (4 reqs), 1 agent (3 reqs)."""
    records = [
        # Human session: 3 requests, gaps of 2000ms and 3000ms
        _make_record("human-s1", "human", 1000, endpoint="/hp/login", status_code=401),
        _make_record("human-s1", "human", 3000, endpoint="/hp/balance"),
        _make_record("human-s1", "human", 6000, endpoint="/hp/history"),
        # Bot session: 4 requests, gaps of 30ms each
        _make_record("bot-s1", "bot", 10000, endpoint="/hp/login"),
        _make_record("bot-s1", "bot", 10030, endpoint="/hp/balance"),
        _make_record("bot-s1", "bot", 10060, endpoint="/hp/login"),
        _make_record("bot-s1", "bot", 10090, endpoint="/hp/balance"),
        # Agent session: 3 requests, gaps of 1500ms and 2000ms
        _make_record("agent-s1", "llm_agent", 20000, endpoint="/hp/login",
                     status_code=401, payload={"user": "a"}),
        _make_record("agent-s1", "llm_agent", 21500, endpoint="/hp/verify",
                     payload={"code": "123"}),
        _make_record("agent-s1", "llm_agent", 23500, endpoint="/hp/balance",
                     payload={"account": "x"}),
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


class TestPipeline:
    def test_produces_csv_with_expected_columns(self, tmp_path):
        """Pipeline output has all 13 expected columns."""
        jsonl_path = tmp_path / "telemetry.jsonl"
        csv_path = tmp_path / "features.csv"
        _write_test_jsonl(jsonl_path)
        result = run_pipeline(jsonl_path, csv_path)
        assert csv_path.exists()
        for col in EXPECTED_FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_one_row_per_session(self, tmp_path):
        """Output has exactly one row per unique session_id."""
        jsonl_path = tmp_path / "telemetry.jsonl"
        csv_path = tmp_path / "features.csv"
        _write_test_jsonl(jsonl_path)

        result = run_pipeline(jsonl_path, csv_path)

        assert len(result) == 3  # human-s1, bot-s1, agent-s1
        assert result["session_id"].nunique() == 3

    def test_all_10_features_present(self, tmp_path):
        """All 10 engineered features exist in output (non-metadata columns)."""
        jsonl_path = tmp_path / "telemetry.jsonl"
        csv_path = tmp_path / "features.csv"
        _write_test_jsonl(jsonl_path)

        result = run_pipeline(jsonl_path, csv_path)

        feature_cols = [
            "mean_gap_time", "std_gap_time", "max_delay", "session_duration",
            "request_rate", "burstiness", "recovery_time", "session_entropy",
            "payload_entropy", "path_repetition",
        ]
        for col in feature_cols:
            assert col in result.columns, f"Missing feature: {col}"

    def test_actor_type_preserved(self, tmp_path):
        """Actor type labels survive the pipeline."""
        jsonl_path = tmp_path / "telemetry.jsonl"
        csv_path = tmp_path / "features.csv"
        _write_test_jsonl(jsonl_path)

        result = run_pipeline(jsonl_path, csv_path)

        assert set(result["actor_type"]) == {"human", "bot", "llm_agent"}

    def test_bot_burstier_than_human(self, tmp_path):
        """Bot burstiness > human burstiness (sanity check on known data)."""
        jsonl_path = tmp_path / "telemetry.jsonl"
        csv_path = tmp_path / "features.csv"
        _write_test_jsonl(jsonl_path)
        result = run_pipeline(jsonl_path, csv_path)
        bot_burst = result.loc[result["actor_type"] == "bot", "burstiness"].iloc[0]
        human_burst = result.loc[result["actor_type"] == "human", "burstiness"].iloc[0]
        assert bot_burst > human_burst

    def test_empty_jsonl_returns_empty(self, tmp_path):
        """Empty JSONL → empty DataFrame, no crash."""
        jsonl_path = tmp_path / "empty.jsonl"
        csv_path = tmp_path / "features.csv"
        jsonl_path.write_text("")

        result = run_pipeline(jsonl_path, csv_path)
        assert result.empty

    def test_csv_matches_returned_df(self, tmp_path):
        """The written CSV matches the returned DataFrame."""
        jsonl_path = tmp_path / "telemetry.jsonl"
        csv_path = tmp_path / "features.csv"
        _write_test_jsonl(jsonl_path)
        result = run_pipeline(jsonl_path, csv_path)
        loaded = pd.read_csv(csv_path)
        assert list(result.columns) == list(loaded.columns)
        assert len(result) == len(loaded)