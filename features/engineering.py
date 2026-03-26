import json
import logging
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

logger = logging.getLogger("honeypot.features")

# Gaps shorter than this (ms) are considered bursty (automated).
# bot 10–50ms, agent 500–3000ms, human 1–8s.
BURST_THRESHOLD_MS = 500.0

# HTTP status codes that count as errors for recovery_time computation.
# 401 = auth failure (login), 403 = forbidden (verify trap).
ERROR_STATUS_CODES = {401, 403}

def compute_timing_features(group: pd.DataFrame) -> dict:
    """Compute timing-based features for a single session.
    Expects rows sorted by timestamp_ms ascending (guaranteed by etl.sort_by_session_and_time).
    Returns a dict of feature names → values. NaN for features that can't
    be computed (e.g., 1-request sessions have no inter-request gaps)."""
    timestamps = group["timestamp_ms"]
    request_count = len(group)

    # Inter-request gaps: diff() computes row[i] - row[i-1].
    # First row is NaN (no predecessor) — dropna() removes it.
    gaps = timestamps.diff().dropna()

    #  mean_gap_time 
    mean_gap = gaps.mean() if len(gaps) > 0 else np.nan

    # std_gap_time 
    # avoid division by zero.
    std_gap = gaps.std() if len(gaps) > 1 else np.nan

    #  max_delay 
    max_delay = gaps.max() if len(gaps) > 0 else np.nan

    #  session_duration 
    duration_ms = timestamps.max() - timestamps.min()

    #  request_rate (requests per second) 
    if duration_ms > 0:
        request_rate = request_count / (duration_ms / 1000.0)
    else:
        request_rate = np.nan  # 1-request session or zero duration

    return {
        "mean_gap_time": mean_gap,
        "std_gap_time": std_gap,
        "max_delay": max_delay,
        "session_duration": float(duration_ms),
        "request_rate": request_rate,
    }


def compute_burstiness(gaps: pd.Series) -> float:
    """Fraction of inter-request gaps shorter than BURST_THRESHOLD_MS.
    Returns 1.0 for fully bursty (all gaps < threshold),
    0.0 for fully deliberate (all gaps >= threshold),
    NaN if no gaps exist."""
    if len(gaps) == 0:
        return np.nan
    return float((gaps < BURST_THRESHOLD_MS).sum() / len(gaps))

def compute_recovery_time(group: pd.DataFrame) -> float:
    """Mean time (ms) between an error response and the actor's next request.
    Finds rows where status_code is 401 or 403, then measures the gap to the
    immediately following row in the session. Uses shift(-1) to look ahead
    one row without a loop. Returns NaN if the session has no error responses.
    """
    timestamps = group["timestamp_ms"].values
    status_codes = group["status_code"].values

    recovery_gaps = []
    for i in range(len(status_codes) - 1):  # -1 because last row has no "next"
        if status_codes[i] in ERROR_STATUS_CODES:
            gap = timestamps[i + 1] - timestamps[i]
            recovery_gaps.append(gap)
    if not recovery_gaps:
        return np.nan
    return float(np.mean(recovery_gaps))

def compute_session_entropy(group: pd.DataFrame) -> float:
    """Shannon entropy (bits) of the endpoint visit distribution.
    High entropy = diverse endpoint visits (goal-driven exploration).
    Low entropy = concentrated on few endpoints (mechanical repetition).
    One unique endpoint → 0.0 (no uncertainty)."""
    counts = group["endpoint"].value_counts()
    probabilities = counts / counts.sum()
    return float(scipy_entropy(probabilities, base=2))

def compute_payload_entropy(group: pd.DataFrame) -> float:
    """Shannon entropy (bits) of unique payload strings across the session.
    Each payload dict is converted to a canonical JSON string (sorted keys)
    for consistent comparison. Null payloads are excluded.
    High entropy = diverse payloads (adaptive behavior).
    Zero entropy = identical payloads (mechanical repetition).
    NaN if all payloads are null."""
    payloads = group["payload"].dropna()
    if len(payloads) == 0:
        return np.nan

    # Canonical form: sorted keys so {"b":1,"a":2} == {"a":2,"b":1}
    canonical = payloads.apply(
        lambda p: json.dumps(p, sort_keys=True, default=str) if isinstance(p, dict) else str(p)
    )

    counts = canonical.value_counts()
    if len(counts) == 0:
        return np.nan

    probabilities = counts / counts.sum()
    return float(scipy_entropy(probabilities, base=2))


def compute_path_repetition(group: pd.DataFrame) -> float:
    """Fraction of consecutive endpoint bigrams that appear more than once.
    Bigrams capture transition patterns, not just frequencies.
    High repetition = cyclic, mechanical traversal (bot-like).
    Low repetition = varied, non-repeating exploration.
    NaN if < 2 requests (no bigrams possible)."""
    endpoints = group["endpoint"].tolist()
    if len(endpoints) < 2:
        return np.nan

    # Build bigrams: pairs of consecutive endpoints
    bigrams = [(endpoints[i], endpoints[i + 1]) for i in range(len(endpoints) - 1)]

    if not bigrams:
        return np.nan

    # Count occurrences of each unique bigram
    bigram_counts = pd.Series(bigrams).value_counts()

    # Repeated bigrams = those appearing more than once
    repeated = (bigram_counts > 1).sum()

    # Fraction of unique bigrams that are repeated
    return float(repeated / len(bigram_counts))

def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master function: transform request-level DataFrame into session-level features.
    Groups by session_id, computes all 10 features per session, attaches
    metadata (session_id, actor_type, request_count).
    Returns a DataFrame with one row per session, 13 columns total."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    grouped = df.groupby("session_id", sort=False)

    for session_id, group in grouped:
        # Timing features (5)
        timing = compute_timing_features(group)

        # Inter-request gaps (recomputed here to pass to burstiness)
        gaps = group["timestamp_ms"].diff().dropna()

        # Burstiness (1)
        burstiness = compute_burstiness(gaps)

        # Recovery time (1)
        recovery = compute_recovery_time(group)

        # Session entropy (1)
        sess_entropy = compute_session_entropy(group)

        # Payload entropy (1)
        pay_entropy = compute_payload_entropy(group)

        # Path repetition (1)
        path_rep = compute_path_repetition(group)

        # Actor type from first row, validated as consistent per session
        # by test_simulation.py (session→actor integrity test)
        actor_type = group["actor_type"].iloc[0]

        rows.append({
            "session_id": session_id,
            "actor_type": actor_type,
            "request_count": len(group),
            **timing,
            "burstiness": burstiness,
            "recovery_time": recovery,
            "session_entropy": sess_entropy,
            "payload_entropy": pay_entropy,
            "path_repetition": path_rep,
        })

    result = pd.DataFrame(rows)

    # Summary
    print(f"[Features] {len(result)} sessions extracted")
    for actor in sorted(result["actor_type"].unique()):
        count = (result["actor_type"] == actor).sum()
        print(f"  {actor}: {count} sessions")

    return result