"""
demo.py — Interactive Multilingual Topic Prediction Demo

Interactive terminal demo for predicting topics from English, Tagalog, or Cebuano input.
Type 'quit' or 'exit' to stop.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict_topic import TopicPredictor


def main():
    print()
    print("=" * 60)
    print("   MULTILINGUAL TOPIC PREDICTOR")
    print("   Supports: English, Tagalog, Cebuano")
    print("=" * 60)
    print()

    # Load model
    predictor = TopicPredictor()

    print("-" * 60)
    print("Type a sentence to predict its topic.")
    print("Type 'quit' or 'exit' to stop.")
    print("-" * 60)
    print()

    while True:
        try:
            text = input("Enter sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not text:
            continue

        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Predict
        result = predictor.predict(text)

        # Display result
        print()
        if result["is_outlier"]:
            print(f"  ⚠  {result['message']}")
            if "suggestion" in result:
                print(f"  💡 Closest topic suggestion: {result['suggestion']}")
        else:
            print(f"  📌 Topic ID:    {result['topic_id']}")
            print(f"  🏷  Topic Name:  {result['topic_label']}")
            print(f"  🔑 Keywords:    {', '.join(result['top_words'])}")

        print()


if __name__ == "__main__":
    main()
