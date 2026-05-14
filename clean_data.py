"""
clean_data.py — Data Cleaning Pipeline for Multilingual Topic Modeling

Cleans original_dataset.csv by:
1. Dropping the 'other' column
2. Dropping rows with blank/NaN in english, tagalog, or cebuano
3. Lowercasing all text
4. Removing special characters and numbers (preserves hyphens for Cebuano)
5. Removing Filipino names (word-level, not substring)
6. Normalizing whitespace
7. Dropping rows that become blank or single-word after cleaning
8. Saving cleaned_data.csv
"""

import pandas as pd
import re
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "original_dataset.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "cleaned_data.csv")
NAMES_FILE = os.path.join(BASE_DIR, "filipino_names.txt")
STOPWORDS_FILE = os.path.join(BASE_DIR, "filipino_stopwords.txt")

LANG_COLUMNS = ["english", "tagalog", "cebuano"]


def load_names(filepath):
    """Load Filipino names list (one name per line, lowercased)."""
    with open(filepath, "r", encoding="utf-8") as f:
        names = set()
        for line in f:
            name = line.strip().lower()
            if name:
                names.add(name)
    print(f"  Loaded {len(names)} names to remove")
    return names


def clean_text(text, names_set):
    """
    Clean a single text string:
    - Lowercase
    - Remove special characters and numbers (keep letters, spaces, hyphens)
    - Remove Filipino names (whole-word match only)
    - Normalize whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove special characters and numbers, keep letters, spaces, hyphens
    # This preserves Cebuano hyphenated words like "kan-on", "mag-aral"
    text = re.sub(r"[^a-záàâãéèêíïóôõúçñü\s\-]", " ", text)

    # Remove standalone hyphens (not part of words)
    text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)

    # Remove names (whole-word match only, so "maria" is removed but "mariano" stays)
    words = text.split()
    words = [w for w in words if w not in names_set]

    # Rejoin and normalize whitespace
    text = " ".join(words).strip()

    return text


def main():
    print("=" * 60)
    print("DATA CLEANING PIPELINE")
    print("=" * 60)

    # ─── Step 1: Load dataset ────────────────────────────────────────────
    print("\n[1/8] Loading dataset...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded: {len(df)} rows, columns: {list(df.columns)}")

    # ─── Step 2: Drop 'other' column ────────────────────────────────────
    print("\n[2/8] Dropping 'other' column...")
    if "other" in df.columns:
        df = df.drop(columns=["other"])
        print("  Dropped 'other' column")
    else:
        print("  'other' column not found, skipping")

    # ─── Step 3: Drop rows with NaN/blank in any language column ─────────
    print("\n[3/8] Dropping rows with missing values...")
    before = len(df)
    df = df.dropna(subset=LANG_COLUMNS)
    after = len(df)
    print(f"  Dropped {before - after} rows with NaN values")
    print(f"  Remaining: {after} rows")

    # Also drop rows where any column is empty string after stripping
    before = len(df)
    for col in LANG_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    mask = df[LANG_COLUMNS].apply(lambda row: all(val != "" for val in row), axis=1)
    df = df[mask].reset_index(drop=True)
    after = len(df)
    print(f"  Dropped {before - after} rows with empty strings")
    print(f"  Remaining: {after} rows")

    # ─── Step 4: Load names list ─────────────────────────────────────────
    print("\n[4/8] Loading Filipino names...")
    names_set = load_names(NAMES_FILE)

    # ─── Step 5: Clean all text columns ──────────────────────────────────
    print("\n[5/8] Cleaning text (lowercase, remove specials/numbers/names)...")
    for col in LANG_COLUMNS:
        df[col] = df[col].apply(lambda x: clean_text(x, names_set))
        print(f"  Cleaned '{col}' column")

    # ─── Step 6: Drop rows that became blank after cleaning ──────────────
    print("\n[6/8] Dropping rows that became blank after cleaning...")
    before = len(df)
    mask = df[LANG_COLUMNS].apply(lambda row: all(val.strip() != "" for val in row), axis=1)
    df = df[mask].reset_index(drop=True)
    after = len(df)
    print(f"  Dropped {before - after} blank rows")
    print(f"  Remaining: {after} rows")

    # ─── Step 7: Drop single-word rows (english column) ──────────────────
    print("\n[7/8] Dropping rows where English column has only 1 word...")
    before = len(df)
    word_counts = df["english"].str.split().apply(len)
    df = df[word_counts > 1].reset_index(drop=True)
    after = len(df)
    print(f"  Dropped {before - after} single-word rows")
    print(f"  Remaining: {after} rows")

    # ─── Step 8: Save cleaned dataset ────────────────────────────────────
    print("\n[8/8] Saving cleaned dataset...")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved to: {OUTPUT_CSV}")

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"  Final rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  English word count stats:")
    wc = df["english"].str.split().apply(len)
    print(f"    Min: {wc.min()}, Max: {wc.max()}, Mean: {wc.mean():.1f}")
    print()

    # Verification checks
    print("VERIFICATION:")
    # No NaN
    nan_count = df[LANG_COLUMNS].isnull().sum().sum()
    print(f"  NaN values: {nan_count} {'✓' if nan_count == 0 else '✗'}")

    # No blank rows
    blank_count = (df[LANG_COLUMNS].apply(lambda row: any(val.strip() == "" for val in row), axis=1)).sum()
    print(f"  Blank rows: {blank_count} {'✓' if blank_count == 0 else '✗'}")

    # No single-word english rows
    single = (df["english"].str.split().apply(len) <= 1).sum()
    print(f"  Single-word english rows: {single} {'✓' if single == 0 else '✗'}")

    # No numbers in english
    has_num = df["english"].apply(lambda x: bool(re.search(r"[0-9]", x))).sum()
    print(f"  English rows with numbers: {has_num} {'✓' if has_num == 0 else '✗'}")

    # No special chars (except hyphens within words)
    has_special = df["english"].apply(lambda x: bool(re.search(r"[^a-záàâãéèêíïóôõúçñü\s\-]", x))).sum()
    print(f"  English rows with special chars: {has_special} {'✓' if has_special == 0 else '✗'}")

    # All lowercase
    is_lower = (df["english"] == df["english"].str.lower()).all()
    print(f"  All lowercase: {'✓' if is_lower else '✗'}")

    # No names present
    names_found = 0
    for name in names_set:
        matches = df["english"].str.split().apply(lambda words: name in words).sum()
        names_found += matches
    print(f"  Filipino names found in english: {names_found} {'✓' if names_found == 0 else '✗'}")

    # No 'other' column
    has_other = "other" in df.columns
    print(f"  'other' column present: {'✗ (still there!)' if has_other else '✓ (removed)'}")

    # Show sample
    print("\nSAMPLE (first 5 rows):")
    print(df.head().to_string(index=False))
    print()


if __name__ == "__main__":
    main()
