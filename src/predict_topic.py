"""
predict_topic.py — Predict the topic of new user input.
"""

from bertopic import BERTopic
from src.config import MODEL_DIR, TOPIC_LABELS_JSON
from src.utils import light_clean, load_topic_labels


class TopicPredictor:
    def __init__(self):
        print("Loading BERTopic model...")
        self.topic_model = BERTopic.load(MODEL_DIR)
        print("Loading topic labels...")
        self.labels = load_topic_labels(TOPIC_LABELS_JSON)
        print("Predictor ready.\n")

    def predict(self, text):
        cleaned = light_clean(text)
        if not cleaned:
            return {"topic_id": -1, "topic_name": "no_topic_outlier", "top_words": [], "message": "Empty input."}

        topics, _ = self.topic_model.transform([cleaned])
        topic_id = topics[0]
        topic_name = self.labels.get(topic_id, f"topic_{topic_id}")

        top_words = []
        if topic_id != -1:
            words = self.topic_model.get_topic(topic_id)
            if words:
                top_words = [w for w, _ in words[:5]]

        result = {"topic_id": topic_id, "topic_name": topic_name, "top_words": top_words}
        if topic_id == -1:
            result["message"] = "No strong topic found. Try a clearer or more specific sentence."
        return result


def main():
    predictor = TopicPredictor()
    test_inputs = [
        "I want to eat rice and chicken.",
        "Gusto kong kumain ng manok.",
        "Gusto nako mokaon og kan-on.",
        "The student is reading a book at school.",
    ]
    for text in test_inputs:
        result = predictor.predict(text)
        print(f"Input: {text}")
        print(f"  Topic ID:   {result['topic_id']}")
        print(f"  Topic Name: {result['topic_name']}")
        print(f"  Top Words:  {', '.join(result['top_words'])}")
        if "message" in result:
            print(f"  Note:       {result['message']}")
        print()


if __name__ == "__main__":
    main()
