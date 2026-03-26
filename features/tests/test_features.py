"""Tests for features/engineering.py — per-feature unit tests with known values."""
import json
import numpy as np
import pandas as pd
import pytest
from features.engineering import (
    BURST_THRESHOLD_MS,
    compute_burstiness,
    compute_path_repetition,
    compute_payload_entropy,
    compute_recovery_time,
    compute_session_entropy,
    compute_timing_features,
)

#  Helpers 

def _make_group(timestamps, **extra_columns):
    """Build a mini-DataFrame representing one session's rows.
    Timestamps are required. Other columns get sensible defaults
    unless overridden via extra_columns."""
    n = len(timestamps)
    data = {
        "timestamp_ms": timestamps,
        "endpoint": extra_columns.pop("endpoint", ["/hp/login"] * n),
        "method": ["POST"] * n,
        "status_code": extra_columns.pop("status_code", [200] * n),
        "payload": extra_columns.pop("payload", [None] * n),
        "session_id": ["test-session"] * n,
        "actor_type": ["human"] * n,
    }
    data.update(extra_columns)
    return pd.DataFrame(data)


#  compute_timing_features 

class TestTimingFeatures:
    def test_mean_gap_time(self):
        """Timestamps [0, 100, 300] -> gaps [100, 200] → mean 150.0"""
        group = _make_group([0, 100, 300])
        result = compute_timing_features(group)
        assert result["mean_gap_time"] == pytest.approx(150.0)

    def test_std_gap_time(self):
        """Timestamps [0, 100, 300] → gaps [100, 200] → std ~70.71"""
        group = _make_group([0, 100, 300])
        result = compute_timing_features(group)
        expected_std = pd.Series([100, 200]).std()  # ddof=1
        assert result["std_gap_time"] == pytest.approx(expected_std)

    def test_max_delay(self):
        """Timestamps [0, 100, 300, 310] → gaps [100, 200, 10] -> max 200"""
        group = _make_group([0, 100, 300, 310])
        result = compute_timing_features(group)
        assert result["max_delay"] == pytest.approx(200.0)

    def test_session_duration(self):
        """Timestamps [100, 500] → duration = 400"""
        group = _make_group([100, 500])
        result = compute_timing_features(group)
        assert result["session_duration"] == pytest.approx(400.0)

    def test_request_rate(self):
        """3 requests over 2000ms → 3 / 2.0 = 1.5 req/s"""
        group = _make_group([0, 1000, 2000])
        result = compute_timing_features(group)
        assert result["request_rate"] == pytest.approx(1.5)

    def test_single_request_returns_nan(self):
        """1-request session -> NaN for all gap features, duration = 0."""
        group = _make_group([5000])
        result = compute_timing_features(group)
        assert pd.isna(result["mean_gap_time"])
        assert pd.isna(result["std_gap_time"])
        assert pd.isna(result["max_delay"])
        assert result["session_duration"] == pytest.approx(0.0)
        assert pd.isna(result["request_rate"])

    def test_two_requests_std_gap_nan(self):
        """2 requests → 1 gap → std needs n>=2 gaps, so NaN."""
        group = _make_group([0, 100])
        result = compute_timing_features(group)
        assert result["mean_gap_time"] == pytest.approx(100.0)
        assert pd.isna(result["std_gap_time"])


#  compute_burstiness 

class TestBurstiness:
    def test_all_gaps_below_threshold(self):
        """All gaps < 500ms → burstiness = 1.0 (fully bursty, bot-like)."""
        gaps = pd.Series([10, 20, 30, 50])
        assert compute_burstiness(gaps) == pytest.approx(1.0)

    def test_all_gaps_above_threshold(self):
        """All gaps >= 500ms → burstiness = 0.0 (fully deliberate, human-like)."""
        gaps = pd.Series([1000, 2000, 5000])
        assert compute_burstiness(gaps) == pytest.approx(0.0)

    def test_mixed_gaps(self):
        """2 of 4 gaps below threshold → burstiness = 0.5."""
        gaps = pd.Series([100, 200, 1000, 2000])
        assert compute_burstiness(gaps) == pytest.approx(0.5)

    def test_no_gaps_returns_nan(self):
        """Empty gaps → NaN."""
        gaps = pd.Series([], dtype=float)
        assert pd.isna(compute_burstiness(gaps))


#  compute_recovery_time 

class TestRecoveryTime:
    def test_recovery_after_401(self):
        """Error at t=100 (401), next request at t=600 → recovery = 500ms."""
        group = _make_group(
            [0, 100, 600, 700],
            status_code=[200, 401, 200, 200],
        )
        result = compute_recovery_time(group)
        assert result == pytest.approx(500.0)

    def test_recovery_after_403(self):
        """Error at t=200 (403), next request at t=1200 → recovery = 1000ms."""
        group = _make_group(
            [0, 200, 1200],
            status_code=[200, 403, 200],
        )
        result = compute_recovery_time(group)
        assert result == pytest.approx(1000.0)

    def test_multiple_errors_averaged(self):
        """Two errors: recovery 500ms and 1000ms → mean = 750ms."""
        group = _make_group(
            [0, 100, 600, 700, 1700],
            status_code=[200, 401, 200, 403, 200],
        )
        # Error at t=100 → next at t=600 → gap 500
        # Error at t=700 → next at t=1700 → gap 1000
        result = compute_recovery_time(group)
        assert result == pytest.approx(750.0)

    def test_no_errors_returns_nan(self):
        """No error status codes → NaN (no recovery to measure)."""
        group = _make_group(
            [0, 100, 200],
            status_code=[200, 200, 200],
        )
        result = compute_recovery_time(group)
        assert pd.isna(result)

    def test_error_on_last_row_ignored(self):
        """Error on last row has no 'next' row → only prior errors counted."""
        group = _make_group(
            [0, 100, 200],
            status_code=[200, 200, 401],  # error on last row
        )
        result = compute_recovery_time(group)
        assert pd.isna(result)  # no recovery gap measurable


#  compute_session_entropy 

class TestSessionEntropy:
    def test_single_endpoint_zero_entropy(self):
        """All requests to one endpoint → entropy = 0.0."""
        group = _make_group(
            [0, 100, 200],
            endpoint=["/hp/login", "/hp/login", "/hp/login"],
        )
        assert compute_session_entropy(group) == pytest.approx(0.0)

    def test_uniform_distribution_max_entropy(self):
        """Equal visits to 4 endpoints → entropy = log2(4) = 2.0."""
        group = _make_group(
            [0, 100, 200, 300],
            endpoint=["/hp/login", "/hp/balance", "/hp/transfer", "/hp/verify"],
        )
        assert compute_session_entropy(group) == pytest.approx(2.0)

    def test_two_endpoints_unequal(self):
        """3 visits to A, 1 visit to B → entropy between 0 and 1."""
        group = _make_group(
            [0, 100, 200, 300],
            endpoint=["/hp/login", "/hp/login", "/hp/login", "/hp/balance"],
        )
        result = compute_session_entropy(group)
        assert 0.0 < result < 1.0


#  compute_payload_entropy 

class TestPayloadEntropy:
    def test_identical_payloads_zero_entropy(self):
        """Same payload repeated -> entropy = 0.0."""
        group = _make_group(
            [0, 100, 200],
            payload=[{"user": "a"}, {"user": "a"}, {"user": "a"}],
        )
        assert compute_payload_entropy(group) == pytest.approx(0.0)

    def test_unique_payloads_positive_entropy(self):
        """All different payloads → entropy > 0."""
        group = _make_group(
            [0, 100, 200],
            payload=[{"user": "a"}, {"user": "b"}, {"user": "c"}],
        )
        result = compute_payload_entropy(group)
        assert result > 0.0

    def test_all_null_payloads_returns_nan(self):
        """All payloads null → NaN."""
        group = _make_group(
            [0, 100, 200],
            payload=[None, None, None],
        )
        result = compute_payload_entropy(group)
        assert pd.isna(result)

    def test_key_order_irrelevant(self):
        """Dicts with same keys in different insertion order → same canonical form → 0.0."""
        group = _make_group(
            [0, 100],
            payload=[{"b": 1, "a": 2}, {"a": 2, "b": 1}],
        )
        assert compute_payload_entropy(group) == pytest.approx(0.0)


#  compute_path_repetition 

class TestPathRepetition:
    def test_cyclic_pattern_high_repetition(self):
        """A→B→A→B → bigrams: (A,B), (B,A), (A,B) → 2 unique, 1 repeated → 0.5."""
        group = _make_group(
            [0, 100, 200, 300],
            endpoint=["/hp/login", "/hp/balance", "/hp/login", "/hp/balance"],
        )
        result = compute_path_repetition(group)
        assert result == pytest.approx(0.5)

    def test_all_unique_bigrams_zero_repetition(self):
        """A→B→C→D → bigrams: (A,B), (B,C), (C,D) → 3 unique, 0 repeated → 0.0."""
        group = _make_group(
            [0, 100, 200, 300],
            endpoint=["/hp/login", "/hp/balance", "/hp/transfer", "/hp/verify"],
        )
        result = compute_path_repetition(group)
        assert result == pytest.approx(0.0)

    def test_single_request_returns_nan(self):
        """< 2 requests → no bigrams → NaN."""
        group = _make_group([0], endpoint=["/hp/login"])
        result = compute_path_repetition(group)
        assert pd.isna(result)

    def test_all_same_endpoint_full_repetition(self):
        """A→A→A→A → bigrams: (A,A), (A,A), (A,A) → 1 unique, 1 repeated → 1.0."""
        group = _make_group(
            [0, 100, 200, 300],
            endpoint=["/hp/login", "/hp/login", "/hp/login", "/hp/login"],
        )
        result = compute_path_repetition(group)
        assert result == pytest.approx(1.0)