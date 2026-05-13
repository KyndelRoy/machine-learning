"""
demo_predict.py — Interactive terminal demo for topic prediction.

Loads the saved model once and enters an interactive loop.
Type a sentence in English, Tagalog, Cebuano, or mixed language.
Type 'quit' or 'exit' to stop.
"""

from src.predict_topic import TopicPredictor


def main():
    print("\n" + "=" * 60)
    print("  MULTILINGUAL TOPIC PREDICTION DEMO")
    print("=" * 60)
    print("  Enter a sentence in English, Tagalog, Cebuano, or mixed.")
    print("  Type 'quit' or 'exit' to stop.\n")

    predictor = TopicPredictor()

    while True:
        try:
            text = input("Enter sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not text:
            print("  (empty input, try again)\n")
            continue

        result = predictor.predict(text)

        print(f"\n  Predicted Topic ID: {result['topic_id']}")
        print(f"  Topic Name:         {result['topic_name']}")
        if result['top_words']:
            print(f"  Top Words:          {', '.join(result['top_words'])}")
        if "message" in result:
            print(f"  Note:               {result['message']}")
        print()


if __name__ == "__main__":
    main()
