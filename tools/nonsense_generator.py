import pandas as pd
import random
import string

# Set seed for reproducibility
random.seed(42)

def generate_nonsense_text():
    """Generates a single row of non-existent/random typed text using varied strategies."""
    strategy = random.choice([
        'random_chars', 'keyboard_smash', 'pseudo_words',
        'mixed_fragments', 'repetitive', 'broken_morphemes'
    ])

    if strategy == 'random_chars':
        length = random.randint(8, 45)
        pool = string.ascii_letters + string.digits + '!@#$%^&*()_+-=[]{}|;:,./<>?'
        return ''.join(random.choices(pool, k=length))

    elif strategy == 'keyboard_smash':
        clusters = ['qwerty', 'asdfgh', 'zxcvbn', 'uiop', 'jkl', 'mnbvc', '12345', '67890']
        num_parts = random.randint(2, 5)
        parts = []
        for _ in range(num_parts):
            cluster = random.choice(clusters)
            length = random.randint(3, 8)
            parts.append(''.join(random.choices(cluster, k=length)))
        return ' '.join(parts)

    elif strategy == 'pseudo_words':
        consonants = 'bcdfghjklmnpqrstvwxyz'
        vowels = 'aeiou'
        num_words = random.randint(3, 7)
        words = []
        for _ in range(num_words):
            word_len = random.randint(2, 6)
            word = ''.join(random.choice(consonants if i % 2 == 0 else vowels) for i in range(word_len))
            words.append(word)
        return ' '.join(words)

    elif strategy == 'mixed_fragments':
        fragments = ['xyz', '123', 'abc', '!!!', '???', 'qwe', 'rty', '789', '...', '###', 'lol', 'brb', 'idk']
        num_frags = random.randint(4, 10)
        return ''.join(random.choices(fragments, k=num_frags))

    elif strategy == 'repetitive':
        base = random.choice(['ha', 'lo', 'br', 'sk', 'zz', 'mm', 'tt', 'aa', 'ee'])
        repeat = random.randint(5, 15)
        return (base * repeat) + random.choice(['!', '?', '.', '', '...'])

    elif strategy == 'broken_morphemes':
        fragments = ['tion', 'ing', 'ment', 'pre', 'sub', 'over', 'under', 'anti', 'pro', 'dis', 'ness', 'ly']
        num_parts = random.randint(3, 6)
        return ''.join(random.choices(fragments, k=num_parts)) + ''.join(random.choices(string.digits, k=random.randint(1, 4)))

# Generate dataset
NUM_ROWS = 2000
print(f"Generating {NUM_ROWS} nonsense/random typed rows...")
nonsense_data = [generate_nonsense_text() for _ in range(NUM_ROWS)]

# Create DataFrame with a single column named 'other' (matches your previous pipeline)
df = pd.DataFrame({'other': nonsense_data})

# Export to CSV
output_file = 'nonexistent_sentences_2000.csv'
df.to_csv(output_file, index=False, encoding='utf-8')

# Verification output
print(f"\nSuccessfully created '{output_file}'")
print(f"Total rows: {len(df)}")
print(f"Column name: '{df.columns[0]}'")
print("\nSample rows:")
print(df.head(10).to_string(index=False))