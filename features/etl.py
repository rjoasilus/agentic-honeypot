import json
import logging
from pathlib import Path
import pandas as pd
logger = logging.getLogger("honeypot.etl")

# validate_schema() checks incoming data against this list.
REQUIRED_COLUMNS = [
    "session_id",
    "timestamp_ms",
    "endpoint",
    "method",
    "payload_size",
    "response_time_ms",
    "ip_hash",
    "user_agent",
    "headers",
    "status_code",
    "actor_type",
    "payload",
    "payload_error",
]

# valid training data.
VALID_ACTOR_TYPES = {"human", "bot", "llm_agent"}

def load_jsonl(path: Path) -> pd.DataFrame:
    """Read a JSONL file into a DataFrame, skipping malformed lines.
    Reads line-by-line with json.loads() so that one corrupted line 
    doesn't kill the entire load. Returns an empty DataFrame with 
    REQUIRED_COLUMNS if the file is empty."""
    path = Path(path)
    if not path.exists():
        logger.warning("JSONL file not found: %s", path)
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    records = []
    skipped = 0

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip blank lines
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
                logger.warning("Skipping malformed JSON on line %d", line_num)

    if skipped > 0:
        logger.info("Loaded %d records, skipped %d malformed lines", len(records), skipped)

    if not records:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    return pd.DataFrame(records)

def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a cleaned copy of the DataFrame. Prints a summary of what
    was dropped and why."""
    if df.empty:
        return df
    initial_count = len(df)

    #  Column presence 
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"JSONL missing required columns: {sorted(missing)}")

    #  Dtype enforcement 
    # pd.to_numeric with errors='coerce' converts unparseable values to NaN
    # instead of crashing. filter those NaN rows out.
    df = df.copy()  # avoid modifying the caller's DataFrame
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df["payload_size"] = pd.to_numeric(df["payload_size"], errors="coerce")
    df["response_time_ms"] = pd.to_numeric(df["response_time_ms"], errors="coerce")
    df["status_code"] = pd.to_numeric(df["status_code"], errors="coerce")

    #  Row filtering 
    # Track drop reasons for the summary printout
    drops = {}

    # Drop rows with null or empty session_id
    mask_session = df["session_id"].isna() | (df["session_id"].astype(str).str.strip() == "")
    drops["null/empty session_id"] = mask_session.sum()
    df = df[~mask_session]

    # Drop rows with null or non-positive timestamp
    mask_time = df["timestamp_ms"].isna() | (df["timestamp_ms"] <= 0)
    drops["invalid timestamp_ms"] = mask_time.sum()
    df = df[~mask_time]

    # Drop rows with unrecognized actor_type
    mask_actor = ~df["actor_type"].isin(VALID_ACTOR_TYPES)
    drops["invalid actor_type"] = mask_actor.sum()
    df = df[~mask_actor]

    #  Cast to final types after filtering 
    df["timestamp_ms"] = df["timestamp_ms"].astype("int64")
    df["payload_size"] = df["payload_size"].astype("int64")
    df["status_code"] = df["status_code"].astype("int64")
    df["response_time_ms"] = df["response_time_ms"].astype("float64")

    #  Summary 
    final_count = len(df)
    total_dropped = initial_count - final_count
    print(f"[ETL] Loaded: {initial_count} rows → Retained: {final_count} rows")
    if total_dropped > 0:
        for reason, count in drops.items():
            if count > 0:
                print(f"  Dropped {count} rows: {reason}")

    return df


def sort_by_session_and_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by session_id then timestamp_ms ascending."""
    if df.empty:
        return df
    return df.sort_values(
        by=["session_id", "timestamp_ms"],
        ascending=[True, True],
    ).reset_index(drop=True)