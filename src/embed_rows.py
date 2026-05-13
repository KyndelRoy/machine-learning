"""
embed_rows.py — Generate multilingual row embeddings.

Steps:
  1. Load cleaned_data.csv
  2. Load the multilingual sentence-transformer model
  3. Encode each language column separately (batch_size=32)
  4. Average available embeddings per row
  5. Save row_embeddings.npy
"""

import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import (
    CLEANED_CSV, EMBEDDINGS_NPY, EMBEDDING_MODEL_NAME,
    BATCH_SIZE, COLUMNS, OUTPUT_DIR,
)
from src.utils import ensure_dirs


def main():
    print("=" * 60)
    print("STEP 2: Generating multilingual embeddings")
    print("=" * 60)

    # Load cleaned data
    df = pd.read_csv(CLEANED_CSV)
    n_rows = len(df)
    print(f"  Loaded {n_rows} rows from cleaned data")

    # Load model
    print(f"  Loading model: {EMBEDDING_MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {embedding_dim}")

    # Encode each language column
    embeddings_by_col = {}
    for col in COLUMNS:
        if col not in df.columns:
            continue

        texts = df[col].fillna("").astype(str).tolist()
        # Track which rows have actual text
        has_text = [bool(t.strip()) for t in texts]

        print(f"\n  Encoding {col} ({sum(has_text)} non-empty sentences)...")
        t0 = time.time()
        emb = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        elapsed = time.time() - t0
        print(f"  {col} encoded in {elapsed:.1f}s")

        # Zero out embeddings for empty/missing texts
        for i, ht in enumerate(has_text):
            if not ht:
                emb[i] = np.zeros(embedding_dim)

        embeddings_by_col[col] = (emb, has_text)

    # Average available embeddings per row
    print("\n  Averaging embeddings per row...")
    row_embeddings = np.zeros((n_rows, embedding_dim), dtype=np.float32)

    for i in range(n_rows):
        embs = []
        for col in COLUMNS:
            if col in embeddings_by_col:
                emb, has_text = embeddings_by_col[col]
                if has_text[i]:
                    embs.append(emb[i])
        if embs:
            row_embeddings[i] = np.mean(embs, axis=0)
        # else remains zero vector (shouldn't happen — english is always present)

    # Save
    ensure_dirs(OUTPUT_DIR)
    np.save(EMBEDDINGS_NPY, row_embeddings)
    print(f"\n  Saved embeddings: shape {row_embeddings.shape}")
    print(f"  File: {EMBEDDINGS_NPY}")
    print()


if __name__ == "__main__":
    main()
