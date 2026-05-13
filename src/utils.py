"""
Shared utility functions for the Multilingual Topic Modeling project.
"""

import os
import re
import json


def light_clean(text):
    """
    Apply minimal text cleaning suitable for transformer embeddings.
    - lowercase
    - strip leading/trailing whitespace
    - collapse multiple spaces into one
    Does NOT remove punctuation or stopwords so transformers keep context.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_topic_labels(path):
    """Load topic labels from a JSON file. Keys are topic IDs (as strings)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Convert string keys back to int
    return {int(k): v for k, v in raw.items()}


def ensure_dirs(*dirs):
    """Create directories if they don't already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
