import pandas as pd

# Load the dataset
df = pd.read_csv('datasets/1-15kcombined_file.csv')

# List of columns to remove
columns_to_remove = ['kapampangan', 'bicolano', 'other']

# Drop the columns if they exist
df_filtered = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

# Save to a new CSV file
df_filtered.to_csv('sample_languages_3.csv', index=False)

import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove emojis and non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove special characters, dots, punctuation, and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# Create a combined column
# We'll join the content of the three columns with a space
df_filtered['combined_text'] = df_filtered['tagalog'].fillna('') + " " + \
                               df_filtered['english'].fillna('') + " " + \
                               df_filtered['cebuano'].fillna('')

# Apply cleaning
df_filtered['cleaned_text'] = df_filtered['combined_text'].apply(clean_text)

# Save to a new CSV file with only the cleaned column
df_filtered[['cleaned_text']].to_csv('cleaned_combined_text.csv', index=False)
print('cleaning done: file name is cleaned_combined_text.csv')

