# bertopic_pipeline.py

import subprocess
import sys
import os
import joblib
import pandas as pd


# ----------------------------
# Auto dependency installer
# ----------------------------
def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        print(f"[INFO] Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


required_packages = [
    "pandas",
    "joblib",
    "bertopic",
    "sentence-transformers",
    "hdbscan",
    "umap-learn",
    "scikit-learn"
]

for pkg in required_packages:
    install_if_missing(pkg)


from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from umap import UMAP


# ----------------------------
# Main pipeline
# ----------------------------
def main():

    df = pd.read_csv("final_cleaned_output.csv")

    # FIX 1: remove NaN FIRST, then convert to list
    df = df.dropna(subset=["combined_text"]).copy()
    texts = df["combined_text"].astype(str).tolist()

    MODEL_PATH = "bertopic_model.pkl"

    if os.path.exists(MODEL_PATH):
        print("[INFO] Loading existing model...")
        topic_model = joblib.load(MODEL_PATH)

        # transform expects List[str]
        topics, probs = topic_model.transform(texts)

    else:
        print("[INFO] Training new model...")

        embedding_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=10,
            min_samples=2,
            prediction_data=True
        )

        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine"
        )

        topic_model = BERTopic(
            embedding_model=embedding_model,
            language="multilingual",
            hdbscan_model=hdbscan_model,
            umap_model=umap_model,
            calculate_probabilities=True,
            verbose=True
        )

        topics, probs = topic_model.fit_transform(texts)

        joblib.dump(topic_model, MODEL_PATH)
        print("[INFO] Model saved.")

    # ----------------------------
    # Results
    # ----------------------------
    topic_info = topic_model.get_topic_info()
    print(topic_info)

    print(topic_model.get_topic(0))

    # FIX 2: no .loc needed anymore
    df["bertopic_topic"] = topics

    df.to_csv("bertopic_output.csv", index=False)
    print("[INFO] Saved output CSV.")

    # ----------------------------
    # Visualizations
    # ----------------------------
    topic_model.visualize_topics()
    topic_model.visualize_barchart()
    topic_model.visualize_heatmap()

    print("[DONE]")


if __name__ == "__main__":
    main()