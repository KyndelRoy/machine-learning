import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords', quiet=True)

INPUT_CSV = "datasets/original_dataset.csv"
OUTPUT_CSV = "datasets/preprocessed_dataset.csv"
OUTPUT_COLUMN = "text"

# STOPWORDS SETUP 

# English
english_stopwords = set(stopwords.words('english'))

shared_ph_stopwords = {
    # names
    "juan", "john", "antonio", "nestor", "daniel", "maria", "tomas", "leo", "yaroslav","tom",
    "mary","name", "called", "english", "tagalog", "cebuano","dina","dana","sina","corina",
    "roy","sarah","jose","maria","cairo","alexandria",

    # Pronouns
    "ako", "ikaw", "siya", "kami", "tayo", "kayo", "mi","kay",
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
    "akin","aking","ako","alin","am","amin","aming","ang",
    "ano","anumang","apat","at","atin","ating","ay","bababa",
    "bago","bakit","bawat","bilang","dahil","dalawa","dapat",
    "din","dito","doon","gagawin","gayunman","ginagawa","ginawa",
    "ginawang","gumawa","gusto","habang","hanggang","hindi","huwag",
    "iba","ibaba","ibabaw","ibig","ikaw","ilagay","ilalim","ilan","inyong",
    "isa","isang","itaas","ito","iyo","iyon","iyong","ka","kahit","kailangan",
    "kailanman","kami","kanila","kanilang","kanino","kanya","kanyang","kapag",
    "kapwa","karamihan","katiyakan","katulad","kaya","kaysa","ko","kong","kulang",
    "kumuha","kung","laban","lahat","lamang","likod","lima","maaari","maaaring",
    "maging","mahusay","makita","marami","marapat","masyado","may","mayroon","mga",
    "minsan","mismo","mula","muli","na","nabanggit","naging","nagkaroon","nais",
    "nakita","namin","napaka","narito","nasaan","ng","ngayon","ni","nila","nilang"
    ,"nito","niya","niyang","noon","o","pa","paano","pababa","paggawa","pagitan",
    "pagkakaroon","pagkatapos","palabas","pamamagitan","panahon","pangalawa","para",
    "paraan","pareho","pataas","pero","pumunta","pumupunta","sa","saan","sabi",
    "sabihin","sarili","sila","sino","siya","tatlo","tayo","tulad","tungkol","una",
    "walang"
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

    text = str(text)
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    # 3. Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    
    # 4. Remove User Mentions and Emails
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    
    # 5. Remove Emojis, Numbers, and Special Characters 
    # [^a-z\s] keeps ONLY lowercase letters and spaces. 
    # This effectively removes all emojis, punctuation, and numbers (standard for topic modeling).
    text = re.sub(r"[^a-z\s]", " ", text)
    
    # 6. Remove extra whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Remove Stopwords
    words = text.split()
    filtered_words = [w for w in words if w not in all_stopwords]

    return " ".join(filtered_words)

def preprocess_pipeline(input_csv, output_csv):
    print(f"Loading input file: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Merge all columns into one column
    # We convert all to string, fill NaN with empty string, and join with a space
    print("Merging all columns into one...")
    df_merged = df.fillna("").astype(str).agg(" ".join, axis=1)
    
    # Create a new dataframe with just the one merged column
    df_new = pd.DataFrame({OUTPUT_COLUMN: df_merged})
    
    print("Cleaning and normalizing text (removing stopwords)...")
    df_new[OUTPUT_COLUMN] = df_new[OUTPUT_COLUMN].apply(clean_text)
    
    # Remove empty/null rows after cleaning
    df_new = df_new[df_new[OUTPUT_COLUMN].str.strip() != ""]
    
    print(f"Saving preprocessed data to {output_csv}")
    df_new.to_csv(output_csv, index=False)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    preprocess_pipeline(INPUT_CSV, OUTPUT_CSV)
