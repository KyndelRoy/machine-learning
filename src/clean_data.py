"""
clean_data.py — Load, clean, and save the dataset.

Steps:
  1. Load original_dataset.csv
  2. Drop the 'other' column
  3. Drop rows where 'english' is NaN
  4. Take only the first MAX_ROWS rows
  5. Apply light cleaning (lowercase, strip, normalize whitespace)
  6. Save outputs/cleaned_data.csv
"""

import pandas as pd
from src.config import (
    RAW_CSV, CLEANED_CSV, COLUMNS, DROP_COLUMNS, MAX_ROWS, OUTPUT_DIR,
)
from src.utils import light_clean, ensure_dirs


def main():
    print("=" * 60)
    print("STEP 1: Cleaning data")
    print("=" * 60)

    # Load
    df = pd.read_csv(RAW_CSV)
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Drop unwanted columns
    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"  Dropped column: {col}")

    # Drop rows with missing English text
    before = len(df)
    df = df.dropna(subset=["english"]).reset_index(drop=True)
    print(f"  Dropped {before - len(df)} rows with missing English text")
    print(f"  Remaining rows: {len(df)}")

    # Take only MAX_ROWS for faster iteration
    if MAX_ROWS and len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS).reset_index(drop=True)
        print(f"  Trimmed to first {MAX_ROWS} rows")

    # Light cleaning on each language column
    for col in COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(light_clean)

    # Replace empty strings with NaN so averaging handles them
    df = df.replace("", pd.NA)

    # Summary
    print(f"\n  Final shape: {df.shape}")
    for col in COLUMNS:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"    {col}: {non_null} non-null")

    # Save
    ensure_dirs(OUTPUT_DIR)
    df.to_csv(CLEANED_CSV, index=False)
    print(f"\n  Saved cleaned data to {CLEANED_CSV}")
    print()


if __name__ == "__main__":
    main()
