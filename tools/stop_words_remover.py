import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')

INPUT_CSV = "cleaned_combined_text.csv"
OUTPUT_CSV = "final_cleaned_output.csv"
TEXT_COLUMN = "cleaned_text"

# STOPWORDS SETUP 

# English
english_stopwords = set(stopwords.words('english'))

shared_ph_stopwords = {
    "ang", "ng", "mga", "sa", "ay", "ako", "ikaw", "siya", "kami", "tayo", "kayo", "mi", "ko",
    "nila", "ito", "iyan", "iyon", "dito", "doon", "kasi", "pero", "kung", "para", 
    "mas", "na", "pa", "ba", "lang", "naman", "pala", "sana", "dahil", "hindi", 
    "ayon", "dal", "an", "um","kundi","gunit","seem", "atoa","tan",
    "ano", "sino", "bakit", "saan", "paano", "lahat", "niya", "ko", "mo", "namin",
    "please", "dont", "alas", "de", "tom", "walang", "taga", "pag", "pwede","kanus", "duha", "every",
    "much","let", "aw", "saako","bang","hapon","adunaye", "good", "maayo", "pagkatapos", "halos", "believe",
    "said", "sinabi", "gawin", "tama", "take", "still", "saw", "ninyo", "itong","anthony",
    "nakita", "importante", "kanunay", "always", "panahon", "time", "oras", "big",
    "dako", "tomorrow", "ugma", "ngayon", "bukas", "pwede", "pag", "iyaha", "kanya",
    "work", "trabaho", "talagang", "tinuod", "hapon", "see", "let", "makita",
    "leave", "taga", "ate", "new", "bago", "bagong", "bang", "kanus", "kailan", 
    "butang", "bagay", "two", "duha", "dalawang", "little", "tingin", "much",
    "aw", "isa", "kada", "nothing", "every", "day", "home", "walang","tulo", "three",
    "years", "happy", "tuig", "stop", "say", "moadto", "going", "went", "pumunta", 
    "beses", "maliit", "small", "alone", "gamay", "pera", "money", "life", "kinabuhi",
    "buhay", "house", "bahay", "balay", "ating", "pang", "atong", "oo", "back",
    "help", "never", "nahitabo", "ginawa", "adunaye", "loob", "nagakaon", "sulod",
    "kumakain", "eat", "kumain", "us", "everything", "aduna", "morning", "umaga", 
    "mali", "una", "sayo", "umalis", "everyone", "old", "call", "well", "mabuti", 
    "magaling", "naging", "afraid", "gumagawa", "lugar", "si", "ga","ka","yan", "iyong",
    "juan", "john", "antonio", "nestor", "daniel", "maria", "tomas", "leo", "yaroslav",
    "mary","name", "called", "english", "tagalog", "cebuano","dina","dana","sina","corina"
}

# Specific to Tagalog (including common particles)
tagalog_stopwords = {
    "mag", "po", "opo", "ho", "daw", "raw", "din", "rin", "eh", "kaya", "lamang","rajod","gina" 
    "ta","nito","nmo","taos", "kayong", "inyong","inyo","kanang", "pagka", "aking","usab","diba"
    "yata", "tulad", "gaya", "habang", "kapag", "kahit", "wala", "meron", "nako", 
    "mong", "kang", "talaga", "may", "yang", "nung", "iyang", "kong", "nang", "mu", "naka",
    "jody", "seemed", "biglang","among", "aming", "giunsa", "gihatag","imo", "imohang",
    "dapat", "kaysa", "akong", "ayaw", "akin", "nag", "anong", "parang", "sinong", 
    "nasaan", "akala", "tapos", "done", "huwag", "ganito", "really", "ingon", "ani", "jud",
    "anyone", "diay", "alanganin", "nato", "natin", "lisud", "lisod", "buong", "araw",
    "pako", "tibouk", "adlaw", "ilang", "pila", "rito", "dere", "diri", "days", "baka", "basin",
    "kitang", "without", "sakon", "alam", "know", "kabalo", "sigurado", "sure", "kahibaw", "lalong", "got",
    "maong", "anything", "koy", "matagal", "long time", "dugay", "think", "murag", "ata", "susunod", "taon", "tuiga", "sunod",
    "next", "year", "hope", "kahapon", "gahapon", "yesterday", "sadyang", "saakong", "nais", "wanted", "gusto",
    "pakiramdam", "feeling", "paminaw", "bihira", "rarely", "usahay", "lunes", "mondays"
}

# Specific to Cebuano / Bisaya
cebuano_stopwords = {
    "ug", "og", "kita", "kamo", "namo", "niini", "kana", "kadto", "dinhi", "didto", 
    "kay", "man", "ra", "lagi", "bitaw", "unta", "karon", "adto", "unsa", 
    "ngano", "naa", "abi", "nakog", "nimo", "imong", "walay", "kinsay", "dili", 
    "usa", "nganong", "akoa", "asa", "sako", "saiyang", "saimong", "gyud", "gi", "tika", "mana",
    "igna", "humana", "saakoa", "ni", "nimong", "tanan", "kalinaw", "sinumang", "bisan", "kinsa",
    "silang", "sila", "kailangang", "kailangan", "need", "pud", "kinahanglan",
    "like", "love", "want", "must", "should", "could", "would", "may", "might", "can", "will", "shall", "no",
    "i", "you", "he", "she", "it", "we", "they", "a", "an", "the", "of", "in", "on", "at", "to", "for",
    "from", "with", "by", "about", "as", "is", "are", "was", "were", "not", "had", "one", "do",
    "go", "gikan", "did", "does", "doin", "didnt", "didn", "kinuha", "nagsulti", 
    "siyang", "looks", "mura", "humingi", "nga", "aron", "wa", "kon", "napakaraming", 
    "many", "daghan", "kaayo", "naay", "nasa", "isang", "kanyang", "iyo", "mao",
    "maraming", "lots", "daghang", "di", "maka", "kog", "lang", "nina", "pero","andong", "yo",
    "unsay", "ginabuhat", "sabihin", "gagawin", "tell", "unsay", "buhaton", "ana",
    "masyadong", "stay", "long", "thought","pirmi", "pirme", "nasud", "igo","saman","siyag","yung","rako",
    "kanila", "uli", "yes", "yun","ta","uy","sige","lage","lagi","giingon", "yata","lng", "maayong",
    "nagti", "mahilig", "ganahan", "ge","tara", "tra","bajud","saimoha","saimo", "ing","sud", "ana"
    "mi", "noong", "katong","nung", "nang"
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