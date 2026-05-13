"""
evaluate_topics.py — Evaluate discovered topic quality using unsupervised metrics.

Metrics:
  - Number of discovered topics
  - Number & percentage of outliers
  - Topic diversity (ratio of unique words across all topics)
  - Topic size distribution (min, max, median, mean)
  - Top words per topic
  - Representative documents per topic
"""

import json
import numpy as np
import pandas as pd
from collections import Counter
from bertopic import BERTopic

from src.config import (
    MODEL_DIR, TOPIC_ASSIGNMENTS_CSV, EVALUATION_REPORT_JSON, OUTPUT_DIR,
)
from src.utils import ensure_dirs


def compute_topic_diversity(topic_model, top_n=10):
    """
    Topic diversity = number of unique words / total words across all topics.
    Higher is better (topics are more distinct).
    """
    all_words = []
    unique_words = set()
    topics = topic_model.get_topic_info()

    for _, row in topics.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        if words:
            for word, _ in words[:top_n]:
                all_words.append(word)
                unique_words.add(word)

    if not all_words:
        return 0.0
    return len(unique_words) / len(all_words)


def main():
    print("=" * 60)
    print("STEP 5: Evaluating topics")
    print("=" * 60)

    # Load model and assignments
    print(f"  Loading model from {MODEL_DIR}...")
    topic_model = BERTopic.load(MODEL_DIR)

    assignments_df = pd.read_csv(TOPIC_ASSIGNMENTS_CSV)
    topics = assignments_df["topic_id"].tolist()

    topic_info = topic_model.get_topic_info()

    # --- Basic counts ---
    n_documents = len(topics)
    n_outliers = topics.count(-1)
    outlier_pct = 100 * n_outliers / n_documents if n_documents > 0 else 0
    n_topics = len(topic_info[topic_info["Topic"] != -1])

    print(f"\n  Documents:        {n_documents}")
    print(f"  Discovered topics: {n_topics}")
    print(f"  Outliers:          {n_outliers} ({outlier_pct:.1f}%)")

    # --- Topic size distribution ---
    topic_counts = Counter(topics)
    # Exclude outlier topic from size stats
    sizes = [count for tid, count in topic_counts.items() if tid != -1]
    size_stats = {
        "min": int(np.min(sizes)) if sizes else 0,
        "max": int(np.max(sizes)) if sizes else 0,
        "median": float(np.median(sizes)) if sizes else 0,
        "mean": float(np.mean(sizes)) if sizes else 0,
    }
    print(f"\n  Topic size distribution:")
    print(f"    Min: {size_stats['min']}, Max: {size_stats['max']}, "
          f"Median: {size_stats['median']:.0f}, Mean: {size_stats['mean']:.1f}")

    # --- Topic diversity ---
    diversity = compute_topic_diversity(topic_model, top_n=10)
    print(f"\n  Topic diversity (top-10): {diversity:.4f}")

    # --- Top words per topic (for report) ---
    top_words_per_topic = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        words = topic_model.get_topic(tid)
        if words:
            top_words_per_topic[str(tid)] = [w for w, _ in words[:10]]

    # --- Representative documents ---
    representative_docs = {}
    try:
        rep_docs = topic_model.get_representative_docs()
        if rep_docs:
            for tid, docs in rep_docs.items():
                if tid == -1:
                    continue
                representative_docs[str(tid)] = docs[:3]
    except Exception:
        pass  # Not all BERTopic versions support this the same way

    # --- Build report ---
    report = {
        "n_documents": n_documents,
        "n_topics": n_topics,
        "n_outliers": n_outliers,
        "outlier_percentage": round(outlier_pct, 2),
        "topic_diversity_top10": round(diversity, 4),
        "topic_size_distribution": size_stats,
        "top_words_per_topic": top_words_per_topic,
        "representative_docs_per_topic": representative_docs,
    }

    # Save
    ensure_dirs(OUTPUT_DIR)
    with open(EVALUATION_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Evaluation report saved to {EVALUATION_REPORT_JSON}")
    print()


if __name__ == "__main__":
    main()
