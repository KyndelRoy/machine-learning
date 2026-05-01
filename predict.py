import joblib

MODEL_PATH = "bertopic_model.pkl"

# load once
topic_model = joblib.load(MODEL_PATH)

def predict_topic(text):
    topics, probs = topic_model.transform([text])

    topic_id = topics[0]
    topic_words = topic_model.get_topic(topic_id)

    return {
        "topic_id": topic_id,
        "keywords": topic_words
    }


if __name__ == "__main__":
    while True:
        user_input = input("Enter text: ")

        result = predict_topic(user_input)

        print("\nTopic ID:", result["topic_id"])
        print("Keywords:", result["keywords"])
        print("-" * 40)