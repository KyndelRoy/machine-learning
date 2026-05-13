"""
generate_topic_labels.py — Generate automatic English keyword-based topic names.

Steps:
  1. Load saved BERTopic model
  2. For each topic (excluding -1): get top N words → join with '_'
  3. Topic -1 → "no_topic_outlier"
  4. Save topic_labels.json
"""

import json
from bertopic import BERTopic

from src.config import MODEL_DIR, TOPIC_LABELS_JSON, TOP_N_WORDS_FOR_LABEL, OUTPUT_DIR
from src.utils import ensure_dirs


def main():
    print("=" * 60)
    print("STEP 4: Generating topic labels")
    print("=" * 60)

    # Load model
    print(f"  Loading model from {MODEL_DIR}...")
    topic_model = BERTopic.load(MODEL_DIR)

    topic_info = topic_model.get_topic_info()
    labels = {}

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]

        if topic_id == -1:
            labels[topic_id] = "no_topic_outlier"
            continue

        # Get top words for this topic
        top_words = topic_model.get_topic(topic_id)
        if top_words:
            word_list = [word for word, _ in top_words[:TOP_N_WORDS_FOR_LABEL]]
            label = "_".join(word_list)
        else:
            label = f"topic_{topic_id}"

        labels[topic_id] = label

    # Save
    ensure_dirs(OUTPUT_DIR)
    with open(TOPIC_LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"  Generated {len(labels)} topic labels")
    print(f"  Saved to {TOPIC_LABELS_JSON}")
    print()

    # Show a few examples
    print("  Sample labels:")
    count = 0
    for tid, label in sorted(labels.items()):
        if tid == -1:
            continue
        print(f"    Topic {tid:>3}: {label}")
        count += 1
        if count >= 10:
            print(f"    ... and {len(labels) - 11} more")
            break
    print()


if __name__ == "__main__":
    main()
