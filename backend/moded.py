import numpy as np
import joblib
from lingua import Language, LanguageDetectorBuilder

# --- Build lingua detector ---
# Only load languages you care about keeping, 
# plus ALL_LANGUAGES to catch everything else
detector = LanguageDetectorBuilder.from_all_languages().build()

# Map lingua's Language enum to your model's class labels
SUPPORTED_LINGUA_LANGUAGES = {
    Language.TAGALOG,
    Language.ENGLISH,
}
# Note: lingua doesn't have Cebuano, Kapampangan, or Bicolano
# That's fine — lingua gates obvious foreign languages,
# your model handles the Philippine language distinctions

import os
# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load your trained model using an absolute path relative to the script
model_path = os.path.join(BASE_DIR, 'language_identifier.pkl')
model = joblib.load(model_path)

CONFIDENCE_THRESHOLD = 0.5

def is_possibly_philippine(text: str) -> bool:
    """
    Returns False only if lingua is highly confident 
    the text is a non-Philippine language.
    """
    confidence_values = detector.compute_language_confidence_values(text)
    
    # Get the top detected language and its confidence
    top = confidence_values[0]  # sorted highest first
    
    NON_PHILIPPINE_LANGUAGES = {
        Language.GERMAN, Language.SPANISH, Language.FRENCH,
        Language.JAPANESE, Language.CHINESE, Language.KOREAN,
        Language.ARABIC, Language.RUSSIAN, Language.PORTUGUESE,
        Language.ITALIAN, Language.DUTCH, Language.MALAY,
        Language.INDONESIAN, Language.VIETNAMESE, Language.THAI,
    }
    
    # If lingua is very confident it's a known foreign language, reject it
    if top.language in NON_PHILIPPINE_LANGUAGES and top.value > 0.8:
        return False
    
    return True  # Uncertain or possibly Philippine — let your model decide


def predict(texts: list[str]) -> list[tuple[str, float]]:
    results = []
    
    # Separate texts that pass the lingua gate
    gate_results = [is_possibly_philippine(t) for t in texts]
    
    # Only run your model on texts that passed the gate
    passed_texts = [t for t, passed in zip(texts, gate_results) if passed]
    
    model_predictions = {}
    if passed_texts:
        proba = model.predict_proba(passed_texts)
        classes = model.classes_
        idx = 0
        for i, passed in enumerate(gate_results):
            if passed:
                probs = proba[idx]
                max_conf = np.max(probs)
                pred = classes[np.argmax(probs)]
                if max_conf < CONFIDENCE_THRESHOLD:
                    model_predictions[i] = ('unrecognized', max_conf)
                else:
                    model_predictions[i] = (pred, max_conf)
                idx += 1

    # Assemble final results
    for i, (text, passed) in enumerate(zip(texts, gate_results)):
        if not passed:
            results.append(('unrecognized', 0.0))
        else:
            results.append(model_predictions[i])

    return results


# --- Test ---
test_sentences = [
    "kung pwede lang dapat pumunta nako don",     # Tagalog
    "Unsay kinahanglan nakong buhaton karon",      # Cebuano
    "Ano an kaipuhan kong gibuhon ngunyan",      # Bicolano
    "Nanu ing kailangan kung daptan ngeni",      # Kapampangan
    "How can I get to the nearest hospital?",    # English
    "Mangan tana king balay na ning koya ku",    # Kapampangan
    "Namumoot ako saimo sa bilog na buhay ko",   # Bicolano
    "Maguuma si Tatay sa bukid kada adlaw",      # Cebuano
    "Gusto ko sana kumain ng masarap na ulam",   # Tagalog
    "The quick brown fox jumps over the dog",    # English
    "Dies ist ein deutscher Satz",               # German (Should be unrecognized)
    "Bonjour, comment ça va aujourd'hui?",        # French (Should be unrecognized)
    "Hola, ¿cómo estás mi querido amigo?",       # Spanish (Should be unrecognized)
    "Watashi wa anata o aishiteimasu",           # Japanese (Should be unrecognized)
    "Selamat pagi lahat, kumusta kayo?",         # Tagalog/Malay mix
    "xkqz mnjp brtl",                            # Gibberish
]

predictions = predict(test_sentences)
print(f"{'Text':<45} {'Detected':<15} {'Confidence'}")
print("-" * 75)
for sent, (lang, conf) in zip(test_sentences, predictions):
    print(f"{sent[:44]:<45} {lang:<15} {conf:.2%}")