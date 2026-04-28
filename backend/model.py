import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib

import os
# 1. Load & Reshape Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '1-10kTag-Eng-Ceb-Kap-Bic.csv')
df = pd.read_csv(csv_path)
df_reshaped = pd.melt(df, value_vars=['cebuano', 'tagalog', 'english', 'kapampangan', 'bicolano'], 
                      var_name='language', value_name='text')
df_reshaped = df_reshaped.dropna()

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df_reshaped['text'], df_reshaped['language'], test_size=0.15, random_state=42
)

# 3. Build Pipeline with Probability Calibration
# LinearSVC doesn't output probabilities natively, so we wrap it with CalibratedClassifierCV
model = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 3))),
    ('clf', CalibratedClassifierCV(LinearSVC(), cv=3, method='sigmoid'))
])

# 4. Train Model
print("Training model...")
model.fit(X_train, y_train)

# 5. Test with Confidence & Unknown Threshold
test_sentences = [
    "If there are enemies what would our thing for now",
    "Anong kailangan kong gawin ngayon",                 # Tagalog
    "Nanu ing kailangan kung daptan ngeni",             # Kapampangan
    "Ano an kaipuhan kong gibuhon ngunyan",             # Bicolano
    "Unsay kinahanglan nakong buhaton karon",           # Cebuano
    "Xylophone quantum blargh 1234567890"               # Expected: Unknown
]

# Get raw probabilities for all known classes
probabilities = model.predict_proba(test_sentences)
classes = model.classes_  # Order matches probability columns

# ⚙️ ADJUST THIS THRESHOLD (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.60

print("\n--- Predictions ---")
for sent, prob in zip(test_sentences, probabilities):
    max_idx = prob.argmax()
    confidence = prob[max_idx]
    predicted_lang = classes[max_idx]
    
    # If confidence is too low, override with "Unknown"
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_lang = "Unknown"
        
    print(f"Text: {sent}")
    print(f"  -> Detected: {predicted_lang} | Confidence: {confidence*100:.2f}%\n")

# 6. Save Model
joblib.dump(model, 'language_identifier.pkl')
print("✅ Model saved as 'language_identifier.pkl'")

# Usage in another script:
# loaded_model = joblib.load('language_identifier.pkl')
# probs = loaded_model.predict_proba(["Kaon na ta!"])[0]
# conf = probs.max()
# lang = loaded_model.classes_[probs.argmax()] if conf >= 0.60 else "Unknown"