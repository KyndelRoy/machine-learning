# import data
import pandas as pd

df = pd.read_csv("cleaned_output.csv")

texts = df["combined_text"].dropna()

#preprocess
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    max_df=0.9,      # ignore overly common words
    min_df=2,        # ignore rare words
    ngram_range=(1,2)  # include bigrams (important for phrases)
)

X = vectorizer.fit_transform(texts)

# train model
from sklearn.decomposition import LatentDirichletAllocation

n_topics = 15  # start with 10

lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42
)

lda.fit(X)

# print topics
def print_topics(model, vectorizer, n_top_words=10):
    words = vectorizer.get_feature_names_out()

    for i, topic in enumerate(model.components_):
        print(f"\nTopic {i}:")
        print(" ".join([words[j] for j in topic.argsort()[-n_top_words:]]))

print_topics(lda, vectorizer)

# Assign topic per row
topic_results = lda.transform(X)

df["topic"] = topic_results.argmax(axis=1)

df.to_csv("lda_output.csv", index=False)