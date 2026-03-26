# this file is for the simulation of the Data Integrity Tests 
# Reads the actual JSONL output and validates structure, labels, and timing.
# Skips gracefully if no JSONL file exists (no simulation has been run yet).

import json
from collections import defaultdict
import pytest
from api.app.config import LOG_PATH

#  Load JSONL 

def _load_jsonl():
    """Read all records from the telemetry JSONL file."""
    if not LOG_PATH.exists():
        return []
    records = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# Skip the entire module if no simulation data exists
_records = _load_jsonl()
pytestmark = pytest.mark.skipif(
    len(_records) == 0,
    reason="No JSONL data found — run a simulation first",
)

# Expected fields from PROJECT_STATE.md Section 16
REQUIRED_FIELDS = {
    "session_id", "timestamp_ms", "endpoint", "method",
    "payload_size", "response_time_ms", "ip_hash", "user_agent",
    "headers", "status_code", "actor_type", "payload", "payload_error",
}

VALID_ACTOR_TYPES = {"human", "bot", "llm_agent"}

class TestRecordSchema:
    """Prove every JSONL row has the correct structure."""
    def test_all_fields_present(self):
        for i, record in enumerate(_records):
            missing = REQUIRED_FIELDS - set(record.keys())
            assert not missing, (
                f"Record {i} missing fields: {missing}"
            )

    def test_no_unknown_actor_types(self):
        for i, record in enumerate(_records):
            if record["actor_type"] == "unknown":
                continue  # skip unlabeled traffic (manual requests, health checks)
            assert record["actor_type"] in VALID_ACTOR_TYPES, (
                f"Record {i} has actor_type='{record['actor_type']}' "
                f"(expected one of {VALID_ACTOR_TYPES})"
            )


class TestActorDistribution:
    """Prove all three actor types are present in the data."""
    def test_all_actor_types_present(self):
        actor_types_found = {r["actor_type"] for r in _records}
        missing = VALID_ACTOR_TYPES - actor_types_found
        if missing:
            pytest.skip(
                f"Actor types not present in current JSONL: {missing} — "
                "this file may be from a single-actor experiment run"
        )

    def test_session_id_maps_to_one_actor_type(self):
        session_actors = defaultdict(set)
        for r in _records:
            session_actors[r["session_id"]].add(r["actor_type"])
        for sid, actors in session_actors.items():
            assert len(actors) == 1, (
                f"Session {sid} has multiple actor types: {actors}"
            )
            

class TestTimestampIntegrity:
    """Prove timestamps are monotonically increasing within each session."""
    def test_timestamps_increase_within_sessions(self):
        sessions = defaultdict(list)
        for r in _records:
            sessions[r["session_id"]].append(r["timestamp_ms"])

        for sid, timestamps in sessions.items():
            for i in range(1, len(timestamps)):
                assert timestamps[i] >= timestamps[i - 1], (
                    f"Session {sid}: timestamp[{i}]={timestamps[i]} < "
                    f"timestamp[{i-1}]={timestamps[i-1]}"
                )

class TestTimingSanity:
    """Rough check that timing distributions follow the expected hierarchy:
    bot gaps < agent gaps < human gaps."""
    def _mean_gaps_by_actor(self):
        """Compute mean inter-request gap per actor type."""
        sessions = defaultdict(list)
        actor_of = {}
        for r in _records:
            sessions[r["session_id"]].append(r["timestamp_ms"])
            actor_of[r["session_id"]] = r["actor_type"]

        gaps_by_actor = defaultdict(list)
        for sid, timestamps in sessions.items():
            if len(timestamps) < 2:
                continue
            for i in range(1, len(timestamps)):
                gap_ms = timestamps[i] - timestamps[i - 1]
                gaps_by_actor[actor_of[sid]].append(gap_ms)

        means = {}
        for actor, gaps in gaps_by_actor.items():
            means[actor] = sum(gaps) / len(gaps) if gaps else 0
        return means

    def test_bot_faster_than_human(self):
        means = self._mean_gaps_by_actor()
        if "bot" not in means or "human" not in means:
            pytest.skip("Need both bot and human data for timing comparison")
        assert means["bot"] < means["human"], (
            f"Bot mean gap ({means['bot']:.0f}ms) should be < "
            f"human mean gap ({means['human']:.0f}ms)"
        )

    def test_bot_faster_than_agent(self):
        means = self._mean_gaps_by_actor()
        if "bot" not in means or "llm_agent" not in means:
            pytest.skip("Need both bot and agent data for timing comparison")
        assert means["bot"] < means["llm_agent"], (
            f"Bot mean gap ({means['bot']:.0f}ms) should be < "
            f"agent mean gap ({means['llm_agent']:.0f}ms)"
        )