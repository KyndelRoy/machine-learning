import pandas as pd

# Load file
df = pd.read_csv('datasets/extended_dataset.csv')

# Reshape from 3 columns to 2 columns (Text and Language)
# This will turn 5,000 rows into 15,000 rows
df_reshaped = pd.melt(df, value_vars=['cebuano', 'tagalog', 'english', 'other'], 
                    var_name='language', value_name='text')

# Drop any empty rows just in case
df_reshaped = df_reshaped.dropna()

print(df_reshaped.head())

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
test_sentence = "erghe qw3rqwe wrywr wrr" 
prediction = model.predict([test_sentence])
print(f"Detected: {prediction[0]}")


import joblib

# Save the model
joblib.dump(model, 'language_identifer.pkl')

# To use it later in a different script:
# loaded_model = joblib.load('language_identifer.pkl')
# loaded_model.predict(["Kaon na ta!"])