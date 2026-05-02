"""
==============================================================================
 Multilingual Topic Classifier — BERTopic
 Supports: Cebuano, Tagalog, English
 Dataset : datasets/original_dataset.csv  (~15,000 parallel rows → ~45,000 entries)
==============================================================================

How it works (high-level):
  1. Load the CSV that has three columns: cebuano, tagalog, english
  2. "Melt" (unpivot) them into a single text column → ~45,000 rows
  3. Light preprocessing (lowercase, remove numbers, collapse whitespace)
  4. Embed every sentence with a multilingual transformer
  5. BERTopic clusters the embeddings and discovers topics
  6. The trained model is saved to disk and can be reloaded later

Usage:
  python model_bertopic.py              # train (first run) or load + predict
  python model_bertopic.py --retrain    # force retrain even if a saved model exists
"""

import os
import sys
import re
import argparse

import pandas as pd
import numpy as np

# ─── Lazy dependency installer (beginner-friendly) ──────────────────────────
def install_if_missing(package, import_name=None):
    """Install a pip package if it is not already importable."""
    import_name = import_name or package
    try:
        __import__(import_name)
    except ImportError:
        import subprocess
        print(f"[INFO] Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

REQUIRED = {
    "bertopic": "bertopic",
    "sentence-transformers": "sentence_transformers",
    "hdbscan": "hdbscan",
    "umap-learn": "umap",
    "scikit-learn": "sklearn",
    "safetensors": "safetensors",
}
for pkg, imp in REQUIRED.items():
    install_if_missing(pkg, imp)

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these values to suit your needs
# ═══════════════════════════════════════════════════════════════════════════════
DATA_PATH       = "datasets/original_dataset.csv"
MODEL_DIR       = "bertopic_multilingual_model"     # folder for the saved model
OUTPUT_CSV      = "bertopic_output.csv"             # labelled dataset output
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Topic modelling hyper-parameters
TARGET_TOPICS    = 15       # aim for 10–20 final topics
MIN_CLUSTER_SIZE = 50       # larger = fewer, broader clusters
MIN_SAMPLES      = 10       # core-point density threshold
UMAP_NEIGHBORS   = 15
UMAP_COMPONENTS  = 5
UMAP_MIN_DIST    = 0.1

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1  ·  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Light preprocessing only:
      - lowercase
      - remove numbers
      - collapse extra whitespace
    We deliberately do NOT remove stopwords because BERTopic's c-TF-IDF
    already down-weights common words and the multilingual model benefits
    from seeing full, natural sentences.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()              # lowercase
    text = re.sub(r"\d+", " ", text)      # remove numbers
    text = re.sub(r"\s+", " ", text)      # collapse whitespace
    return text.strip()


def load_and_expand_dataset(csv_path: str) -> list[str]:
    """
    Load the parallel corpus and expand it from ~15k rows × 3 columns
    into a single list of ~45k cleaned sentences.
    """
    print(f"[INFO] Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"       Rows: {len(df)}, Columns: {list(df.columns)}")

    # Melt (unpivot) the three language columns into one column
    #   Before:  tagalog | english | cebuano   (15,000 rows)
    #   After :  text                          (45,000 rows)
    df_melted = df.melt(
        value_vars=["cebuano", "tagalog", "english"],
        var_name="language",
        value_name="text",
    )

    # Drop any NaN / empty rows
    df_melted = df_melted.dropna(subset=["text"])
    df_melted = df_melted[df_melted["text"].astype(str).str.strip() != ""]

    # Clean
    df_melted["text"] = df_melted["text"].apply(clean_text)

    # Remove duplicates that may appear after cleaning
    df_melted = df_melted.drop_duplicates(subset=["text"])

    texts = df_melted["text"].tolist()
    print(f"[INFO] Expanded dataset: {len(texts)} unique sentences")
    return texts


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2  ·  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(texts: list[str], embedding_model) -> BERTopic:
    """
    Build and train a BERTopic model on the provided texts.

    Pipeline:
      Sentence Embeddings  →  UMAP (reduce dimensions)  →  HDBSCAN (cluster)
      →  c-TF-IDF (topic representation)  →  Topic Reduction
    """

    # --- Dimensionality reduction ---
    umap_model = UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        n_components=UMAP_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )

    # --- Clustering ---
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        prediction_data=True,            # needed for .transform() later
        cluster_selection_method="eom",   # Excess of Mass → more fine-grained
    )

    # --- Vectorizer (for c-TF-IDF keyword extraction) ---
    # We keep stopwords because the user asked NOT to aggressively remove them.
    # min_df=5 filters out very rare words that add noise.
    vectorizer_model = CountVectorizer(
        min_df=5,
        ngram_range=(1, 2),    # unigrams + bigrams for richer topic labels
    )

    # --- BERTopic ---
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="multilingual",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,      # so we get confidence scores
        verbose=True,
    )

    # Fit
    print("[INFO] Fitting BERTopic model (this may take a few minutes) ...")
    topics, probs = topic_model.fit_transform(texts)

    initial_count = len(set(topics)) - (1 if -1 in topics else 0)
    outlier_count = topics.count(-1)
    print(f"[INFO] Initial topics found : {initial_count}")
    print(f"[INFO] Outlier documents    : {outlier_count} / {len(texts)}")

    # --- Topic Reduction ---
    if initial_count > TARGET_TOPICS:
        print(f"[INFO] Reducing to ~{TARGET_TOPICS} topics ...")
        topic_model.reduce_topics(texts, nr_topics=TARGET_TOPICS)
        reduced_count = len(set(topic_model.topics_)) - (1 if -1 in topic_model.topics_ else 0)
        print(f"[INFO] Topics after reduction: {reduced_count}")

    # --- Reduce outliers (reassign -1 docs to nearest topic) ---
    print("[INFO] Reducing outlier assignments ...")
    new_topics = topic_model.reduce_outliers(texts, topic_model.topics_)
    topic_model.update_topics(texts, topics=new_topics)

    final_outliers = new_topics.count(-1) if isinstance(new_topics, list) else int((np.array(new_topics) == -1).sum())
    print(f"[INFO] Remaining outliers   : {final_outliers}")

    return topic_model


def save_model(topic_model: BERTopic, model_dir: str):
    """Save the trained model using BERTopic's native serialization."""
    print(f"[INFO] Saving model to {model_dir}/ ...")
    topic_model.save(
        model_dir,
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=EMBEDDING_MODEL,
    )
    print("[INFO] Model saved successfully.")


def save_labelled_csv(texts: list[str], topic_model: BERTopic, output_path: str):
    """Save the dataset with its assigned topic labels to a CSV file."""
    df_out = pd.DataFrame({
        "text": texts,
        "topic_id": topic_model.topics_,
    })
    # Add topic label (top keywords) for readability
    topic_info = topic_model.get_topic_info()
    id_to_name = dict(zip(topic_info["Topic"], topic_info["Name"]))
    df_out["topic_label"] = df_out["topic_id"].map(id_to_name)
    df_out.to_csv(output_path, index=False)
    print(f"[INFO] Labelled dataset saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3  ·  PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def predict_topic(text: str, topic_model: BERTopic) -> dict:
    """
    Predict the topic of a single text input.

    Parameters
    ----------
    text : str
        A sentence or paragraph in Cebuano, Tagalog, or English.
    topic_model : BERTopic
        A fitted (or loaded) BERTopic model.

    Returns
    -------
    dict with keys:
        topic_id    – int     (the cluster id, -1 = outlier)
        topic_label – str     (human-readable topic name)
        keywords    – list    (top representative words)
        confidence  – float   (highest topic probability, 0–1)
    """
    cleaned = clean_text(text)

    # transform() returns (list[int], ndarray)
    topics, probs = topic_model.transform([cleaned])
    topic_id = topics[0]

    # Gather topic keywords
    if topic_id == -1:
        keywords = []
        label = "Outlier / Unclassified"
    else:
        topic_words = topic_model.get_topic(topic_id)          # list of (word, score)
        keywords = [word for word, _ in topic_words[:10]]      # top-10 keywords
        topic_info = topic_model.get_topic_info()
        match = topic_info[topic_info["Topic"] == topic_id]
        label = match["Name"].values[0] if len(match) > 0 else f"Topic {topic_id}"

    # Confidence: highest probability across all topics
    if probs is not None and len(probs) > 0:
        confidence = float(np.max(probs[0]))
    else:
        confidence = 0.0

    return {
        "topic_id": topic_id,
        "topic_label": label,
        "keywords": keywords,
        "confidence": confidence,
    }


def print_prediction(result: dict):
    """Pretty-print a prediction result."""
    print(f"  📌 Topic ID    : {result['topic_id']}")
    print(f"  📝 Topic Label : {result['topic_label']}")
    print(f"  🔑 Keywords    : {', '.join(result['keywords']) if result['keywords'] else 'N/A'}")
    print(f"  📊 Confidence  : {result['confidence']:.2%}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4  ·  SAMPLE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_INPUTS = [
    # English
    "The government plans to build a new highway connecting the two provinces.",
    "Students are struggling to attend online classes because of slow internet.",
    "The price of rice has increased significantly this month.",
    # Tagalog
    "Nagplano ang gobyerno na magtayo ng bagong ospital sa probinsya.",
    "Ang mga mag-aaral ay nahihirapan sa online na klase dahil sa mabagal na internet.",
    # Cebuano
    "Ang gobyerno nagplano og pagtukod ug bag-ong ospital sa probinsya.",
    "Ang mga estudyante naglisod sa online nga klase tungod sa hinay nga internet.",
]


def run_sample_predictions(topic_model: BERTopic):
    """Run and display predictions on a handful of example sentences."""
    print("\n" + "=" * 60)
    print("📋 SAMPLE PREDICTIONS")
    print("=" * 60)

    for text in SAMPLE_INPUTS:
        print(f"\n  Input: \"{text}\"")
        result = predict_topic(text, topic_model)
        print_prediction(result)
        print("  " + "-" * 50)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5  ·  INTERACTIVE LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_loop(topic_model: BERTopic):
    """Let the user type sentences and get real-time topic predictions."""
    print("\n" + "=" * 60)
    print("🧠 TOPIC PREDICTOR — INTERACTIVE MODE")
    print("   Type a sentence in English, Tagalog, or Cebuano.")
    print("   Type 'topics' to see all discovered topics.")
    print("   Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting ...")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye! 👋")
            break
        if user_input.lower() == "topics":
            print("\n" + topic_model.get_topic_info().to_string() + "\n")
            continue

        result = predict_topic(user_input, topic_model)
        print()
        print_prediction(result)
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multilingual BERTopic Classifier")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain even if a saved model exists")
    args = parser.parse_args()

    # ── Load the embedding model (always needed) ──
    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # ── Train or Load ──
    model_exists = os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR)

    if model_exists and not args.retrain:
        # --- LOAD existing model ---
        print(f"[INFO] Found saved model in {MODEL_DIR}/ — loading ...")
        topic_model = BERTopic.load(MODEL_DIR, embedding_model=embedding_model)
        print(f"[INFO] Model loaded. Topics: {len(topic_model.get_topic_info()) - 1}")

    else:
        # --- TRAIN new model ---
        if args.retrain:
            print("[INFO] --retrain flag set. Retraining from scratch ...")

        texts = load_and_expand_dataset(DATA_PATH)
        topic_model = train_model(texts, embedding_model)

        # Save model to disk
        save_model(topic_model, MODEL_DIR)

        # Save labelled CSV
        save_labelled_csv(texts, topic_model, OUTPUT_CSV)

    # ── Show topic overview ──
    print("\n" + "=" * 60)
    print("📊 DISCOVERED TOPICS")
    print("=" * 60)
    topic_info = topic_model.get_topic_info()
    # Print all topics except -1 (outlier)
    display = topic_info[topic_info["Topic"] != -1][["Topic", "Count", "Name"]].head(20)
    print(display.to_string(index=False))

    # ── Sample predictions ──
    run_sample_predictions(topic_model)

    # ── Interactive mode ──
    interactive_loop(topic_model)


if __name__ == "__main__":
    main()