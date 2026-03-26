"""Recover tarpit sessions from combined telemetry.jsonl.
Filters out Phase 5 session IDs and builds features_tarpit.csv
from the remaining (tarpit) sessions only.
"""
import pandas as pd
from api.app.config import PROJECT_ROOT
from features.etl import load_jsonl, validate_schema, sort_by_session_and_time
from features.engineering import build_session_features

# Load the original Phase 5 session IDs
original = pd.read_csv(PROJECT_ROOT / "data/processed/features.csv")
original_ids = set(original["session_id"])
print(f"Phase 5 sessions: {len(original_ids)}")

# Load full JSONL (Phase 5 + tarpit combined)
df = load_jsonl(PROJECT_ROOT / "data/raw/telemetry.jsonl")
df = validate_schema(df)
df = sort_by_session_and_time(df)
print(f"Total JSONL rows: {len(df)}")

# Keep only rows whose session_id is NOT in Phase 5
tarpit_df = df[~df["session_id"].isin(original_ids)]
print(f"Tarpit rows: {len(tarpit_df)}")

# Session counts by actor type
counts = tarpit_df.groupby("actor_type")["session_id"].nunique()
print(f"Tarpit sessions by actor:")
for actor, count in counts.items():
    print(f"  {actor}: {count}")

# Build features for tarpit sessions only
features = build_session_features(tarpit_df)
out = PROJECT_ROOT / "data/processed/features_tarpit.csv"
features.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(f"Shape: {features.shape}")
