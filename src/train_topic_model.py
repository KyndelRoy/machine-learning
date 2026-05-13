"""
train_topic_model.py — Train a BERTopic model.

Steps:
  1. Load English documents from cleaned_data.csv
  2. Load pre-computed row_embeddings.npy
  3. Configure and train BERTopic
  4. Save the trained model
  5. Save topic_assignments.csv and topic_info.csv
"""

import time
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from src.config import (
    CLEANED_CSV, EMBEDDINGS_NPY, MODEL_DIR,
    TOPIC_ASSIGNMENTS_CSV, TOPIC_INFO_CSV,
    EMBEDDING_MODEL_NAME, MIN_TOPIC_SIZE,
    CALCULATE_PROBABILITIES, LANGUAGE, RANDOM_SEED, OUTPUT_DIR,
)
from src.utils import ensure_dirs


def main():
    print("=" * 60)
    print("STEP 3: Training BERTopic model")
    print("=" * 60)

    # Load documents and embeddings
    df = pd.read_csv(CLEANED_CSV)
    english_docs = df["english"].fillna("").astype(str).tolist()
    print(f"  Loaded {len(english_docs)} English documents")

    embeddings = np.load(EMBEDDINGS_NPY)
    print(f"  Loaded embeddings: shape {embeddings.shape}")

    assert len(english_docs) == embeddings.shape[0], (
        f"Document count ({len(english_docs)}) != embedding rows ({embeddings.shape[0]})"
    )

    # Load the same embedding model so BERTopic can use it for transform/predict
    print(f"  Loading embedding model for BERTopic: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Create BERTopic
    print(f"  Creating BERTopic (min_topic_size={MIN_TOPIC_SIZE})...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language=LANGUAGE,
        min_topic_size=MIN_TOPIC_SIZE,
        calculate_probabilities=CALCULATE_PROBABILITIES,
        verbose=True,
    )

    # Train
    print("  Training...")
    t0 = time.time()
    topics, _ = topic_model.fit_transform(english_docs, embeddings=embeddings)
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.1f}s")

    # Summary
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info["Topic"] != -1])
    n_outliers = topics.count(-1)
    print(f"\n  Discovered {n_topics} topics")
    print(f"  Outliers (Topic -1): {n_outliers} ({100 * n_outliers / len(topics):.1f}%)")

    # Save model
    ensure_dirs(MODEL_DIR)
    topic_model.save(MODEL_DIR, serialization="safetensors", save_ctfidf=True, save_embedding_model=EMBEDDING_MODEL_NAME)
    print(f"  Model saved to {MODEL_DIR}")

    # Save topic assignments
    ensure_dirs(OUTPUT_DIR)
    assignments_df = pd.DataFrame({
        "document": english_docs,
        "topic_id": topics,
    })
    assignments_df.to_csv(TOPIC_ASSIGNMENTS_CSV, index=False)
    print(f"  Topic assignments saved to {TOPIC_ASSIGNMENTS_CSV}")

    # Save topic info
    topic_info.to_csv(TOPIC_INFO_CSV, index=False)
    print(f"  Topic info saved to {TOPIC_INFO_CSV}")
    print()


if __name__ == "__main__":
    main()
