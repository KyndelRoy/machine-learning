"""
predict_topic.py — Topic Prediction for Multilingual Input

Loads the saved BERTopic model and predicts topics for new input.
Supports English, Tagalog, and Cebuano sentences.

Usage:
    from predict_topic import TopicPredictor
    predictor = TopicPredictor()
    result = predictor.predict("I want to eat chicken")
"""

import json
import os
import re
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
LABELS_PATH = os.path.join(BASE_DIR, "output", "topic_labels.json")
NAMES_FILE = os.path.join(BASE_DIR, "filipino_names.txt")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_names(filepath):
    """Load Filipino names list for cleaning."""
    names = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip().lower()
                if name:
                    names.add(name)
    return names


def clean_input(text, names_set):
    """
    Apply the same cleaning as training:
    - Lowercase
    - Remove special characters and numbers
    - Remove Filipino names (whole word)
    - Normalize whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"[^a-záàâãéèêíïóôõúçñü\s\-]", " ", text)
    text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)

    words = text.split()
    words = [w for w in words if w not in names_set]

    return " ".join(words).strip()


class TopicPredictor:
    """Loads a saved BERTopic model and predicts topics for new input."""

    def __init__(self):
        """Load model, embedding model, and topic labels."""
        print("Loading topic prediction model...")

        # Load names for cleaning
        self.names_set = load_names(NAMES_FILE)

        # Load topic labels
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            self.topic_labels = json.load(f)

        # Load embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Load BERTopic model
        self.topic_model = BERTopic.load(MODEL_DIR, embedding_model=self.embedding_model)

        print("Model loaded successfully.")
        print(f"  Topics available: {len(self.topic_labels) - 1}")  # -1 for topic -1
        print()

    def predict(self, text):
        """
        Predict the topic for a given sentence.

        Args:
            text: Input sentence (English, Tagalog, or Cebuano)

        Returns:
            dict with topic_id, topic_label, top_words, is_outlier
        """
        # Clean input the same way as training data
        cleaned = clean_input(text, self.names_set)

        if not cleaned:
            return {
                "input": text,
                "cleaned": "",
                "topic_id": -1,
                "topic_label": "no_strong_topic",
                "top_words": [],
                "is_outlier": True,
                "message": "Input was empty after cleaning.",
            }

        # Generate embedding
        embedding = self.embedding_model.encode([cleaned])

        # Predict topic
        topics, probs = self.topic_model.transform([cleaned], embeddings=embedding)
        topic_id = topics[0]

        # Get topic info
        is_outlier = topic_id == -1
        topic_label = self.topic_labels.get(str(topic_id), f"topic_{topic_id}")

        # Get top words
        top_words = []
        if not is_outlier:
            words = self.topic_model.get_topic(topic_id)
            if words:
                top_words = [w for w, _ in words[:5]]

        # If outlier, suggest closest topic
        suggestion = None
        if is_outlier:
            # Find nearest non-outlier topic by trying to get approximate topic
            all_topics = self.topic_model.get_topic_info()
            non_outlier = all_topics[all_topics["Topic"] != -1]
            if len(non_outlier) > 0:
                # Use the model's built-in approximate method
                topics_approx, _ = self.topic_model.transform([cleaned], embeddings=embedding)
                if topics_approx[0] != -1:
                    suggestion = self.topic_labels.get(str(topics_approx[0]), f"topic_{topics_approx[0]}")

        result = {
            "input": text,
            "cleaned": cleaned,
            "topic_id": int(topic_id),
            "topic_label": topic_label,
            "top_words": top_words,
            "is_outlier": is_outlier,
        }

        if suggestion:
            result["suggestion"] = suggestion

        if is_outlier:
            result["message"] = "No strong topic found. Try a clearer or more specific sentence."

        return result
