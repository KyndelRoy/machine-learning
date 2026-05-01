import stopwordsiso
import string

def clean_and_filter(text, lang_code):
    # 1. Remove punctuation
    text_no_punct = text.translate(str.maketrans('', '', string.punctuation))
    
    # 2. Tokenize and lowercase
    words = text_no_punct.lower().split()
    
    # 3. Load stopwords for the specific language
    stop_words = stopwordsiso.stopwords(lang_code)
    
    # 4. Filter
    filtered = [w for w in words if w not in stop_words]
    
    return filtered

# --- TEST 1: ENGLISH ---
en_input = "The quick brown fox jumps over the lazy dog in the morning."
en_output = clean_and_filter(en_input, "en")

# --- TEST 2: TAGALOG ---
tl_input = "Ang mabilis na kulay kayumangging loro ay lumilipad sa ibabaw ng mga puno."
tl_output = clean_and_filter(tl_input, "tl")

# --- DISPLAY RESULTS ---
print("--- ENGLISH SAMPLE ---")
print(f"Input:  {en_input}")
print(f"Output: {en_output}")

print("\n--- TAGALOG SAMPLE ---")
print(f"Input:  {tl_input}")
print(f"Output: {tl_output}")