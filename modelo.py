# ==============================================================================
# PHASE 1: DATA LOADING & ENVIRONMENT SETUP
# ==============================================================================
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
import stopwordsiso 

# Load your parallel dataset
# Assuming columns: 'english', 'tagalog', 'cebuano'
df = pd.read_csv('datasets/original_dataset.csv')

# ==============================================================================
# PHASE 2: CUSTOM STOPWORD PREPARATION
# ==============================================================================
# We need these for Phase 4 to ensure topic names are meaningful.
tl_stopwords = list(stopwordsiso.stopwords("tl"))
ceb_stopwords = [
    "ang", "mga", "sa", "ug", "nga", "ni", "si", "kay", "man", "ba", "gyud",
    # Pronouns
    "kita", "kamo", "namo", "kog", "nimo", "imong", "akoa",
    "siyang", "sila", "silang", "kanila",
    "mao", "iyo", "kanyang",
    "sako", "saiyang", "saimong",
    "ta", "rako", "siyag", "saimo",

    # Core Cebuano particles / markers
    "ug", "og", "ra", "lagi", "lage", "bitaw", "unta", "karon",
    "man", "nga", "aron", "wa", "kon", "gi", "gyud", "pud",
    "kaayo", "uy", "sige", "ge", "tara", "tra", "bajud",
    "naay", "ana", "lng", "yata",

    # Demonstratives / locatives
    "niini", "kana", "kadto", "dinhi", "didto",
    "adto", "gikan", "katong",

    # Question words
    "unsa", "unsay", "ngano", "nganong", "kinsay", "asa",

    # Negation / existential
    "dili", "di", "walay",

    # Other particles
    "maka", "abi", "nakog", "tika", "mana",
    "igna", "humana", "ni", "nimong",
    "tanan", "bisan", "kinsa",
    "kailangan", "kinahanglan", "need",
    "igo", "basin",

    # Tagalog particles common in Cebuano code-switched text
    "nasa", "isang", "pero", "lang", "nang", "nung", "yung",
    "nina", "nag",

    # English function words (code-switching is heavy in PH text)
    "i", "you", "he", "she", "it", "we", "they",
    "a", "an", "the",
    "of", "in", "on", "at", "to", "for", "from",
    "with", "by", "about", "as",
    "is", "are", "was", "were", "did", "does", "do",
    "not", "no", "yes", "had", "didnt", "didn",

    # Shared cross-list
    "mi", "yo", "usa", "yun", "uli",

    # names
    "juan", "john", "antonio", "nestor", "daniel", "maria", "tomas", "leo", "yaroslav","tom",
    "mary","name", "called", "english", "tagalog", "cebuano","dina","dana","sina","corina","roy","sarah","jose","maria",
    "cairo","alexandria",

    # Pronouns
    "ako", "ikaw", "siya", "kami", "tayo", "kayo", "mi","kay"
    "ko", "mo", "nila", "niya", "namin", "ninyo",
    "ito", "iyan", "iyon", "itong",
    "si", "sina", "iyong", "iyaha", "kanya", "ka", "yan",
    "ating", "atong", "atoa",

    # Core particles / markers
    "ang", "ng", "mga", "sa", "ay",
    "na", "pa", "ba", "lang", "naman", "pala", "sana",
    "kasi", "pero", "kung", "para", "mas", "dahil", "hindi",
    "kundi", "ayon", "bang", "pang", "pag", "taga", "an",

    # Question words (cross-language)
    "ano", "sino", "bakit", "saan", "paano", "kanus", "kailan",

    # Existential / negation
    "walang", "aduna", "oo",

    # High-frequency universal quantifiers / determiners
    "lahat", "isa", "kada",

    # Modal
    "pwede",

    # Temporal (extremely high-frequency, non-topical)
    "ngayon"
    ]
en_stopwords = list(stopwordsiso.stopwords("en"))

# Combine them all for a robust multilingual vectorizer
combined_stopwords = tl_stopwords + ceb_stopwords + en_stopwords

# ==============================================================================
# PHASE 3: BUILDING THE MULTILINGUAL PIPELINE
# ==============================================================================

# 1. Choose a Multilingual Embedding Model
# This model "understands" 50+ languages and maps them to the same vector space.
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 2. Configure the Vectorizer (This handles the "Phase 2" stopword removal)
# This ensures that "ang" and "the" don't become the topic names.
vectorizer_model = CountVectorizer(stop_words=combined_stopwords)

# 3. Fine-tune Topic Representations (Optional but recommended for showcase)
representation_model = KeyBERTInspired()

# 4. Initialize BERTopic
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    verbose=True # Shows progress bar
)

# ==============================================================================
# PHASE 4: TRAINING & TOPIC EXTRACTION
# ==============================================================================

# Note: You can feed all three columns into the model. 
# This helps the model see more "context" for the same concepts.
# Clean data: Combine columns, drop nulls, and ensure all entries are strings to avoid ValueError
all_text = pd.concat([df['english'], df['tagalog'], df['cebuano']]).dropna().astype(str).tolist()

# Fit the model
topics, probs = topic_model.fit_transform(all_text)

# ==============================================================================
# PHASE 5: SAVING & SHOWCASING RESULTS
# ==============================================================================

# Get the list of topics generated
topic_info = topic_model.get_topic_info()
print(topic_info.head(10))

# Save the topic list to CSV for your report
topic_info.to_csv("topic_results.csv", index=False)

# VISUALIZATIONS (Perfect for your presentation)
# 1. Visualize Topics in 2D space
fig_clusters = topic_model.visualize_topics()
fig_clusters.write_html("topic_clusters.html")

# 2. Visualize Term Rank (Word importance per topic)
fig_hierarchy = topic_model.visualize_hierarchy()
fig_hierarchy.write_html("topic_hierarchy.html")

# 3. Visualize Barchart of Top Words
fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
fig_barchart.write_html("topic_barchart.html")

print("Analysis Complete. HTML visualizations saved to folder.")