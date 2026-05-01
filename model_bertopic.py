import subprocess
import sys
import os
import pandas as pd

# ----------------------------
# Auto dependency installer
# ----------------------------
def install_if_missing(package, import_name=None):
    import_name = import_name or package
    try:
        __import__(import_name)
    except ImportError:
        print(f"[INFO] Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Note: installed names vs import names
required_packages = {
    "pandas": "pandas", 
    "joblib": "joblib", 
    "bertopic": "bertopic", 
    "sentence-transformers": "sentence_transformers", 
    "hdbscan": "hdbscan", 
    "umap-learn": "umap", 
    "scikit-learn": "sklearn"
}

for pip_name, import_name in required_packages.items():
    install_if_missing(pip_name, import_name)

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from umap import UMAP

def main():
    # CONFIG
    DATA_PATH = "datasets/final_cleaned_output_15k.csv"
    MODEL_DIR = "bertopic_multilingual_model" # Using a directory for native saving
    TARGET_TOPICS = 50
    MIN_CLUSTER_SIZE = 40
    
    # 1. Initialize the embedding model 
    # (Needs to be loaded regardless of whether we are training or predicting)
    print("[INFO] Loading Embedding Model...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # 2. Check if we need to train or just load
    if os.path.exists(MODEL_DIR):
        print(f"[INFO] Loading existing model from {MODEL_DIR}...")
        # Load natively, passing the embedding model back in
        topic_model = BERTopic.load(MODEL_DIR, embedding_model=embedding_model)
        
    else:
        print("[INFO] No existing model found. Starting training pipeline...")
        
        # Load and prep data
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=["combined_text"]).copy()
        texts = df["combined_text"].astype(str).tolist()

        # Initialize dimensionality reduction
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.1,
            metric="cosine",
            random_state=42
        )

        # Initialize clustering (Bug fixed: removed random_state)
        hdbscan_model = HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=5,
            prediction_data=True,
            cluster_selection_method='eom'
        )

        # Initialize BERTopic
        topic_model = BERTopic(
            embedding_model=embedding_model,
            language="multilingual",
            hdbscan_model=hdbscan_model,
            umap_model=umap_model,
            calculate_probabilities=True,
            verbose=True
        )

        # Fit model
        print("[INFO] Fitting model...")
        topics, probs = topic_model.fit_transform(texts)
        print(f"[INFO] Initial topics found: {len(topic_model.get_topic_info())}")

        # Reduce topics
        print(f"[INFO] Reducing to ~{TARGET_TOPICS} topics...")
        topic_model.reduce_topics(texts, nr_topics=TARGET_TOPICS)

        # Save model natively using safetensors
        print("[INFO] Saving model...")
        topic_model.save(MODEL_DIR, serialization="safetensors", save_ctfidf=True)
        
        # Save output CSV
        df["bertopic_topic"] = topic_model.topics_
        df.to_csv("bertopic_output.csv", index=False)
        print("[INFO] Training complete and dataset saved with topics.")

    # ----------------------------
    # Results Overview
    # ----------------------------
    print("\n--- TOPIC OVERVIEW ---")
    print(topic_model.get_topic_info().head(10)) # Print top 10 topics

    # ----------------------------
    # Real-Time User Prediction Logic
    # ----------------------------
    print("\n" + "="*50)
    print("🧠 TOPIC PREDICTOR READY")
    print("Type a sentence in English, Tagalog, or Cebuano to predict its topic.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        user_input = input("\nEnter text to predict: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
            
        if not user_input.strip():
            continue

        # Predict the topic for the new input
        predicted_topics, probabilities = topic_model.transform([user_input])
        top_topic = predicted_topics[0]
        
        if top_topic == -1:
            print(">> Prediction: Outlier / No specific topic found (Topic -1)")
        else:
            # Get the keywords representing the topic
            topic_words = topic_model.get_topic(top_topic)
            keywords = ", ".join([word[0] for word in topic_words[:5]])
            
            print(f">> Predicted Topic ID: {top_topic}")
            print(f">> Topic Keywords: {keywords}")
            # If probabilities were calculated/saved, print the confidence
            if probabilities is not None and len(probabilities) > 0:
                 print(f">> Confidence: {max(probabilities[0]):.2%}")

if __name__ == "__main__":
    main()