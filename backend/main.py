import pandas as pd

import os
# Load your translation dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'combined_languages_6.csv')
df = pd.read_csv(csv_path)

# Reshape from wide to long format (Text and Language)
# This will stack all language columns into a single column
df_reshaped = pd.melt(df, value_vars=['cebuano', 'tagalog', 'english', 'kapampangan', 'bicolano', 'other'], 
                    var_name='language', value_name='text')

# Drop any empty rows just in case
df_reshaped = df_reshaped.dropna()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# 1. Split the reshaped data
X_train, X_test, y_train, y_test = train_test_split(
    df_reshaped['text'], df_reshaped['language'], test_size=0.15, random_state=42
)

# 2. Build a pipeline with a Character-level Vectorizer
# 'char' is better than 'word' for distinguishing Tagalog vs Cebuano
model = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 3))),
    ('clf', LinearSVC())
])

# 3. Train
model.fit(X_train, y_train)

# 4. Quick Test
test_sentences = [
    "If there are enemies what would our thing for now",
    "kung pwede lang dapat pumunta nako don", # Tagalog
    "Nanu ing kailangan kung daptan ngeni", # Kapampangan (Sample)
    "Ano an kaipuhan kong gibuhon ngunyan", # Bicolano (Sample)
    "Unsay kinahanglan nakong buhaton karon", # Cebuano
    "ngano naa ka diri bai dili man ta friends",
    "Mangan tana kening balay na ning koya ku",
    "Dies ist ein deutscher Satz",               # German
    "Bonjour, comment ça va aujourd'hui?",        # French
    "Hola, ¿cómo estás mi querido amigo?",       # Spanish
    "Watashi wa anata o aishiteimasu",           # Japanese
    "Привет, как дела?",                         # Russian
    "Grabe yung traffic kahapon, I was so late for my meeting", # Taglish
    "Kaon na ta sa school, I'm so hungry na jud",               # Bislish
    "Actually, kailangan ko na talaga mag-start ng project na ito", # Taglish
    "Mangan tana but please make sure the food is clean",        # Kapampangan + English
    "Gusto nako mokaon but wala man koy kwarta",                 # Cebuano + Tagalog mix
    "12431iuh89 fddkwb 233232 efkj234 f",
    "xkqz mnjp brtl",
    "xkqz mnjp brtl i dont know feasufhgui",
]
predictions = model.predict(test_sentences)
for sent, pred in zip(test_sentences, predictions):
    print(f"Detected: {pred} | Text: {sent}")

import joblib

# Save the model
joblib.dump(model, 'language_identifer.pkl')

# To use it later in a different script:
# loaded_model = joblib.load('language_identifer.pkl')
# loaded_model.predict(["Kaon na ta!"])