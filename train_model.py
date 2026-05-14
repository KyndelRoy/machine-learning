"""
train_model.py — BERTopic Training Pipeline for Multilingual Topic Modeling

Pipeline:
  Phase 1 (Clustering): Embeddings → UMAP → HDBSCAN
  Phase 2 (Representation): CountVectorizer → c-TF-IDF → topic keywords
  Post-processing: Outlier reduction, label generation, evaluation

Trains on the English column only. The multilingual embedding model
(paraphrase-multilingual-MiniLM-L12-v2) maps all languages into the same
vector space, so topics learned from English work for Tagalog/Cebuano input.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "cleaned_data.csv")
STOPWORDS_FILE = os.path.join(BASE_DIR, "filipino_stopwords.txt")
NAMES_FILE = os.path.join(BASE_DIR, "filipino_names.txt")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_TOPIC_SIZE = 15
RANDOM_SEED = 42


def load_stopwords():
    """Load combined English + Filipino stopwords for CountVectorizer."""
    # English stopwords (sklearn default list)
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stopwords = set(ENGLISH_STOP_WORDS)

    # Filipino stopwords
    if os.path.exists(STOPWORDS_FILE):
        with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    stopwords.add(word)
        print(f"  Loaded Filipino stopwords from {os.path.basename(STOPWORDS_FILE)}")

    # Also add names as stopwords (belt-and-suspenders with clean_data.py)
    if os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    stopwords.add(word)
        print(f"  Loaded Filipino names from {os.path.basename(NAMES_FILE)}")

    print(f"  Total stopwords: {len(stopwords)}")
    return list(stopwords)


def main():
    print("=" * 60)
    print("BERTOPIC TRAINING PIPELINE")
    print("=" * 60)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ─── Step 1: Load cleaned data ──────────────────────────────────────
    print("\n[1/7] Loading cleaned data...")
    df = pd.read_csv(INPUT_CSV)
    docs = df["english"].tolist()
    print(f"  Loaded {len(docs)} documents")

    # ─── Step 2: Load stopwords ─────────────────────────────────────────
    print("\n[2/7] Loading stopwords...")
    stopwords = load_stopwords()

    # ─── Step 3: Generate embeddings ────────────────────────────────────
    embeddings_path = os.path.join(OUTPUT_DIR, "embeddings.npy")

    if os.path.exists(embeddings_path):
        print("\n[3/7] Loading cached embeddings...")
        embeddings = np.load(embeddings_path)
        print(f"  Loaded embeddings: shape {embeddings.shape}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    else:
        print("\n[3/7] Generating embeddings (this may take a few minutes on CPU)...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings = embedding_model.encode(
            docs,
            show_progress_bar=True,
            batch_size=32,
        )
        np.save(embeddings_path, embeddings)
        print(f"  Saved embeddings: shape {embeddings.shape}")

    # ─── Step 4: Configure BERTopic components ─────────────────────────
    print("\n[4/7] Configuring BERTopic...")

    # Phase 1: Clustering components
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_SEED,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Phase 2: Representation components
    vectorizer_model = CountVectorizer(
        stop_words=stopwords,
        min_df=2,
        ngram_range=(1, 2),  # Include bigrams for better topic labels
    )

    print(f"  min_topic_size: {MIN_TOPIC_SIZE}")
    print(f"  embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"  ngram_range: (1, 2)")

    # ─── Step 5: Train BERTopic ─────────────────────────────────────────
    print("\n[5/7] Training BERTopic model...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=MIN_TOPIC_SIZE,
        calculate_probabilities=False,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    # Topic stats
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info["Topic"] != -1])
    n_outliers = (np.array(topics) == -1).sum()
    outlier_pct = n_outliers / len(topics) * 100

    print(f"\n  Topics discovered: {n_topics}")
    print(f"  Outliers: {n_outliers} ({outlier_pct:.1f}%)")

    # ─── Step 5.5: Outlier reduction if needed ──────────────────────────
    if outlier_pct > 20:
        print(f"\n  Outlier rate > 20%, applying outlier reduction...")
        topics = topic_model.reduce_outliers(docs, topics)
        topic_model.update_topics(docs, topics=topics, vectorizer_model=vectorizer_model)
        n_outliers_after = (np.array(topics) == -1).sum()
        print(f"  Outliers after reduction: {n_outliers_after} ({n_outliers_after / len(topics) * 100:.1f}%)")

    # ─── Step 6: Generate topic labels & save outputs ───────────────────
    print("\n[6/7] Generating topic labels and saving outputs...")

    # Get updated topic info
    topic_info = topic_model.get_topic_info()

    # Generate keyword-based labels
    topic_labels = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            topic_labels[str(tid)] = "no_strong_topic"
            continue
        # Get top words for this topic
        top_words = topic_model.get_topic(tid)
        if top_words:
            label = "_".join([w for w, _ in top_words[:4]])
            topic_labels[str(tid)] = label
        else:
            topic_labels[str(tid)] = f"topic_{tid}"

    # Save topic labels
    labels_path = os.path.join(OUTPUT_DIR, "topic_labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(topic_labels, f, indent=2)
    print(f"  Saved topic labels: {labels_path}")

    # Save topic info
    info_path = os.path.join(OUTPUT_DIR, "topic_info.csv")
    topic_info.to_csv(info_path, index=False)
    print(f"  Saved topic info: {info_path}")

    # Save data with topic assignments
    df["topic_id"] = topics
    df["topic_label"] = df["topic_id"].apply(lambda x: topic_labels.get(str(x), f"topic_{x}"))
    data_path = os.path.join(OUTPUT_DIR, "data_with_topics.csv")
    df.to_csv(data_path, index=False)
    print(f"  Saved data with topics: {data_path}")

    # Save the BERTopic model
    topic_model.save(MODEL_DIR, serialization="safetensors", save_ctfidf=True, save_embedding_model=EMBEDDING_MODEL_NAME)
    print(f"  Saved model: {MODEL_DIR}")

    # ─── Step 7: Evaluation ─────────────────────────────────────────────
    print("\n[7/7] Evaluating topic quality...")

    # Topic count and size distribution
    topic_sizes = topic_info[topic_info["Topic"] != -1]["Count"].tolist()

    # Topic diversity (ratio of unique words across all topics)
    all_topic_words = set()
    per_topic_words = []
    for tid in topic_info[topic_info["Topic"] != -1]["Topic"]:
        words = [w for w, _ in topic_model.get_topic(tid)]
        per_topic_words.append(words)
        all_topic_words.update(words)

    total_words = sum(len(w) for w in per_topic_words)
    diversity = len(all_topic_words) / total_words if total_words > 0 else 0

    evaluation = {
        "num_topics": n_topics,
        "num_documents": len(docs),
        "num_outliers": int((np.array(topics) == -1).sum()),
        "outlier_percentage": round((np.array(topics) == -1).sum() / len(topics) * 100, 2),
        "topic_diversity": round(diversity, 4),
        "min_topic_size_setting": MIN_TOPIC_SIZE,
        "avg_topic_size": round(np.mean(topic_sizes), 1) if topic_sizes else 0,
        "median_topic_size": round(np.median(topic_sizes), 1) if topic_sizes else 0,
        "max_topic_size": max(topic_sizes) if topic_sizes else 0,
        "min_topic_size_actual": min(topic_sizes) if topic_sizes else 0,
        "embedding_model": EMBEDDING_MODEL_NAME,
    }

    eval_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)
    print(f"  Saved evaluation report: {eval_path}")

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Documents: {len(docs)}")
    print(f"  Topics discovered: {n_topics}")
    print(f"  Outliers: {evaluation['num_outliers']} ({evaluation['outlier_percentage']}%)")
    print(f"  Topic diversity: {evaluation['topic_diversity']}")
    print(f"  Avg topic size: {evaluation['avg_topic_size']}")

    print("\n  Top 10 topics:")
    top_topics = topic_info[topic_info["Topic"] != -1].head(10)
    for _, row in top_topics.iterrows():
        tid = row["Topic"]
        label = topic_labels.get(str(tid), f"topic_{tid}")
        print(f"    Topic {tid}: {label} (size: {row['Count']})")

    print(f"\n  Saved files:")
    print(f"    Model:         {MODEL_DIR}/")
    print(f"    Topic labels:  {labels_path}")
    print(f"    Topic info:    {info_path}")
    print(f"    Data+topics:   {data_path}")
    print(f"    Embeddings:    {embeddings_path}")
    print(f"    Evaluation:    {eval_path}")
    print()


if __name__ == "__main__":
    main()
