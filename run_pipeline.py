"""
run_pipeline.py — Run the full training pipeline end-to-end.

Pipeline:
  1. Clean data
  2. Generate row embeddings
  3. Train BERTopic model
  4. Generate topic labels
  5. Evaluate topics
"""

import time


def main():
    print("\n" + "=" * 60)
    print("  MULTILINGUAL TOPIC MODELING PIPELINE")
    print("=" * 60 + "\n")

    overall_start = time.time()

    from src.clean_data import main as clean_data
    from src.embed_rows import main as embed_rows
    from src.train_topic_model import main as train_model
    from src.generate_topic_labels import main as generate_labels
    from src.evaluate_topics import main as evaluate

    steps = [
        ("Clean data", clean_data),
        ("Generate embeddings", embed_rows),
        ("Train BERTopic", train_model),
        ("Generate topic labels", generate_labels),
        ("Evaluate topics", evaluate),
    ]

    for name, func in steps:
        t0 = time.time()
        func()
        elapsed = time.time() - t0
        print(f"  [{name}] completed in {elapsed:.1f}s\n")

    total = time.time() - overall_start
    print("=" * 60)
    print(f"  PIPELINE COMPLETE — Total time: {total:.1f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
