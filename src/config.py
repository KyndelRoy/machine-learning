"""
Central configuration for the Multilingual Topic Modeling project.
All paths, model settings, and hyperparameters are defined here.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input
RAW_CSV = os.path.join(PROJECT_ROOT, "original_dataset.csv")

# Output directory
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CLEANED_CSV = os.path.join(OUTPUT_DIR, "cleaned_data.csv")
EMBEDDINGS_NPY = os.path.join(OUTPUT_DIR, "row_embeddings.npy")
TOPIC_ASSIGNMENTS_CSV = os.path.join(OUTPUT_DIR, "topic_assignments.csv")
TOPIC_INFO_CSV = os.path.join(OUTPUT_DIR, "topic_info.csv")
TOPIC_LABELS_JSON = os.path.join(OUTPUT_DIR, "topic_labels.json")
EVALUATION_REPORT_JSON = os.path.join(OUTPUT_DIR, "evaluation_report.json")
SAMPLE_PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "sample_predictions.csv")

# Model directory
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "topic_model")

# ──────────────────────────────────────────────
# Dataset settings
# ──────────────────────────────────────────────
COLUMNS = ["english", "tagalog", "cebuano"]
DROP_COLUMNS = ["other"]
MAX_ROWS = 5000  # Use a subset for faster initial iteration

# ──────────────────────────────────────────────
# Embedding model
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 32

# ──────────────────────────────────────────────
# BERTopic settings
# ──────────────────────────────────────────────
MIN_TOPIC_SIZE = 10
CALCULATE_PROBABILITIES = False
LANGUAGE = "english"
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Topic label settings
# ──────────────────────────────────────────────
TOP_N_WORDS_FOR_LABEL = 4
