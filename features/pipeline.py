import argparse
from pathlib import Path
import pandas as pd
from api.app.config import LOG_PATH, PROJECT_ROOT
from features.etl import load_jsonl, validate_schema, sort_by_session_and_time
from features.engineering import build_session_features

# Default output path: data/processed/features.csv
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "features.csv"

def run_pipeline(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Execute the full ETL -> feature engineering -> CSV pipeline.
    Returns the feature DataFrame for programmatic use
    Also writes it to output_path as CSV."""
    print(f"\n{'*'*60}")
    print("Data Pipeline + Feature Engineering")
    print(f"{'*'*60}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}\n")

    # Load 
    print("[1/4] Loading JSONL...")
    df = load_jsonl(input_path)
    if df.empty:
        print("  No data found. Pipeline stopped.")
        return df

    # Validate 
    print("\n[2/4] Validating schema...")
    df = validate_schema(df)
    if df.empty:
        print("  No valid rows after cleaning. Pipeline stopped.")
        return df

    # Sort 
    print("\n[3/4] Sorting by session and timestamp...")
    df = sort_by_session_and_time(df)

    # Feature Engineering 
    print("\n[4/4] Engineering session-level features...")
    features_df = build_session_features(df)
    if features_df.empty:
        print("  No sessions produced. Pipeline stopped.")
        return features_df

    # Write CSV 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print(f"\n{'*'*80}")
    print(f"Pipeline complete. Output: {output_path}")
    print(f"  Shape: {features_df.shape[0]} sessions × {features_df.shape[1]} columns")
    print(f"{'*'*80}\n")

    return features_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the ETL + feature engineering pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=LOG_PATH,
        help=f"Path to raw JSONL telemetry file (default: {LOG_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for output CSV (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    run_pipeline(args.input, args.output)