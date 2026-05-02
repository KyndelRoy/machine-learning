import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')

INPUT_CSV = "datasets/cleaned_combined_text_15k.csv"
OUTPUT_CSV = "datasets/final_cleaned_output_15k.csv"
TEXT_COLUMN = "cleaned_text"

# STOPWORDS SETUP 

# English
english_stopwords = set(stopwords.words('english'))

shared_ph_stopwords = {
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
    "ngayon",
}

tagalog_stopwords = {
    # Particles unique to Tagalog
    "mag", "po", "opo", "ho", "daw", "raw", "din", "rin", "eh",
    "kaya", "lamang", "nito",
    "yata", "tulad", "gaya", "habang", "kapag", "kahit",
    "wala", "meron", "nako",
    "mong", "kang", "may", "yang", "nung", "iyang", "kong",
    "nang", "mu", "naka", "nag",

    # Conjunctions / discourse markers
    "diba", "tapos", "huwag", "ganito",
    "baka", "lalong", "parang", "kaysa", "dapat",

    # Pronouns / possessives
    "aming", "imo", "imohang",
    "akong", "akin", "ayaw",
    "nato", "natin", "saakong", "kitang", "sakon",

    # Question words (Tagalog-specific)
    "anong", "sinong", "nasaan",

    # Cebuano particles frequently mixed in Tagalog text
    "giunsa", "ani", "jud", "diay",
    "koy", "murag", "ata",

    # Others
    "anyone", "anything",
    "ilang", "pila",
    "rito", "diri",
    "noong", "ta",
    "usab",
}

cebuano_stopwords = {
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
}

# Combine all
all_stopwords = (
    english_stopwords
    .union(shared_ph_stopwords)
    .union(tagalog_stopwords)
    .union(cebuano_stopwords)
)

# CLEANING FUNCTION

def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)

    words = text.split()
    filtered_words = [w for w in words if w not in all_stopwords]

    return " ".join(filtered_words)

#  PROCESS CSV

df = pd.read_csv(INPUT_CSV)

df["combined_text"] = df[TEXT_COLUMN].apply(clean_text)

# Remove empty/null rows
df = df[df["combined_text"].str.strip() != ""]


df[["combined_text"]].to_csv(OUTPUT_CSV, index=False)
print(f"Cleaned file saved as: {OUTPUT_CSV}")