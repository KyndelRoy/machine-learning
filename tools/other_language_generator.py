import pandas as pd
from faker import Faker
import random

# Set seeds for reproducibility
random.seed(42)
Faker.seed(42)

# Define all 18 language locales (including Southeast Asian)
locales = [
    'fr_FR', # French
    'es_ES', # Spanish
    'de_DE', # German
    'it_IT', # Italian
    'pt_BR', # Portuguese
    'ru_RU', # Russian
    'nl_NL', # Dutch
    'pl_PL', # Polish
    'tr_TR', # Turkish
    'sv_SE', # Swedish
    'cs_CZ', # Czech
    'ja_JP', # Japanese
    'zh_CN', # Chinese
    'ar_SA', # Arabic
    'hi_IN', # Hindi
    'id_ID', # Indonesian
    'vi_VN', # Vietnamese
    'th_TH'  # Thai
]

SENTENCES_PER_LANG = 1000
all_sentences = []

print("Generating sentences...")
for loc in locales:
    print(f"  Generating {SENTENCES_PER_LANG} sentences for {loc}...")
    fake = Faker(loc)
    lang_sentences = [fake.sentence() for _ in range(SENTENCES_PER_LANG)]
    all_sentences.extend(lang_sentences)

# Shuffle all sentences to prevent batch bias during testing
random.shuffle(all_sentences)

# Create DataFrame with a single column named 'other'
df = pd.DataFrame({'other': all_sentences})

# Export to CSV
output_file = 'multilingual_other.csv'
df.to_csv(output_file, index=False, encoding='utf-8')

# Verification output
print(f"\nSuccessfully created '{output_file}'")
print(f"Total rows: {len(df)}")
print(f"Languages: {len(locales)} x {SENTENCES_PER_LANG} = {len(df)} rows")
print(f"Column name: '{df.columns[0]}'")
print("\nFirst 10 rows:")
print(df.head(50).to_string(index=False))