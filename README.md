# Multilingual Topic Modeling Guide

This project trains a Python model for topic prediction using parallel text in:

- Tagalog
- English
- Cebuano

Each row is expected to contain one sentence in these 3 languages. The script supports two workflows:

1. Supervised topic classification
2. Unsupervised topic discovery

Use supervised classification if your dataset has a topic label column such as `topic`.
Use unsupervised discovery if your dataset only has the 3 text columns and no label yet.

## Files

- `multilingual_topic_model.py` - main training and prediction script
- `requirements_topic_model.txt` - Python dependencies
- `clean_dataset.csv` - sample dataset with 3 text columns
- `extended_dataset.csv` - extended sample dataset
- `filipino_stopwords.txt` - optional stopwords file

## Dataset Format

### Option 1: Labeled dataset for topic prediction

Your CSV should look like this:

```csv
tagalog,english,cebuano,topic
mahilig ako sa basketball,i like basketball,ganahan ko og basketball,sports
nag aaral ako ng algebra,i study algebra,nagtuon ko og algebra,education
```

Required columns:

- `tagalog`
- `english`
- `cebuano`
- topic label column such as `topic`

### Option 2: Unlabeled dataset for topic discovery

Your CSV can also look like this:

```csv
tagalog,english,cebuano
mahilig ako sa basketball,i like basketball,ganahan ko og basketball
nag aaral ako ng algebra,i study algebra,nagtuon ko og algebra
```

In this case, the script will group similar rows into discovered topics, but it will not know human-readable labels unless you assign them later.

## Installation

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements_topic_model.txt
```

## How The Model Works

The script combines the Tagalog, English, and Cebuano sentence from each row into one training sample.

For supervised classification, it uses:

- TF-IDF word features
- TF-IDF character features
- `LinearSVC` classifier

This is a strong baseline for text classification on a dataset around 5,000 rows.

For unlabeled topic discovery, it uses:

- TF-IDF features
- `NMF` topic modeling

## Using Stopwords

You can optionally pass stopword files for Tagalog or Cebuano.

Example:

```bash
--stopword-files filipino_stopwords.txt cebuano_stopwords.txt
```

Each stopword file should contain one stopword per line.

## Run The Script

### 1. Automatic mode

This mode checks whether a label column exists.

- If a label column exists, it trains a classifier.
- If no label column exists, it runs topic discovery.

```bash
python3 multilingual_topic_model.py \
  --input-csv clean_dataset.csv \
  --task auto \
  --text-columns tagalog english cebuano \
  --stopword-files filipino_stopwords.txt
```

### 2. Supervised topic classification

Use this when your CSV has a topic label column.

```bash
python3 multilingual_topic_model.py \
  --input-csv your_labeled_data.csv \
  --task classify \
  --text-columns tagalog english cebuano \
  --label-column topic \
  --stopword-files filipino_stopwords.txt
```

### 3. Unsupervised topic discovery

Use this when your CSV has only the 3 text columns.

```bash
python3 multilingual_topic_model.py \
  --input-csv clean_dataset.csv \
  --task discover \
  --text-columns tagalog english cebuano \
  --stopword-files filipino_stopwords.txt \
  --num-topics 8 \
  --top-words 
```

### 4. Predict topics on new data

After training a supervised classifier, use the saved model for prediction:

```bash
python3 multilingual_topic_model.py \
  --task predict \
  --input-csv new_rows.csv \
  --text-columns tagalog english cebuano \
  --model-path outputs/topic_classifier.joblib
```

## Output Files

The script saves results into the `outputs/` folder.

### For classification

- `outputs/topic_classifier.joblib` - saved trained classifier
- `outputs/classification_predictions.csv` - actual vs predicted labels
- `outputs/classification_metrics.json` - accuracy, F1, label stats

### For topic discovery

- `outputs/topic_discovery_model.joblib` - saved discovery model
- `outputs/topic_discovery_assignments.csv` - topic assignment per row
- `outputs/topic_discovery_summary.csv` - keywords for each topic

### For prediction

- `outputs/predictions.csv` - predicted topics for new rows

## Important Notes

### If your current CSV has only 3 columns

If your CSV only contains:

- `tagalog`
- `english`
- `cebuano`

then the model cannot learn exact topic names yet.

It can only:

- discover topic clusters
- assign topic IDs
- show top keywords per topic

If you want real topic prediction like `sports`, `politics`, or `education`, you need a labeled column such as `topic`.

## Example Labeled Topics

You can create a `topic` column with labels like:

- `sports`
- `education`
- `health`
- `politics`
- `technology`
- `food`
- `travel`
- `entertainment`

Then train with `--task classify`.

## Tips For Better Results

- Keep labels consistent. Use `sports` everywhere instead of mixing `sport` and `sports`.
- Clean obvious spelling noise if possible.
- Add more labeled rows if one topic has too few examples.
- Use your Tagalog and Cebuano stopwords if they improve signal.
- Start with supervised classification if you already know your target topics.

## Quick Start

If you already have a labeled CSV:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_topic_model.txt
python3 multilingual_topic_model.py \
  --input-csv your_labeled_data.csv \
  --task classify \
  --text-columns tagalog english cebuano \
  --label-column topic
```

If you only have the 3 text columns:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_topic_model.txt
python3 multilingual_topic_model.py \
  --input-csv clean_dataset.csv \
  --task discover \
  --text-columns tagalog english cebuano \
  --num-topics 8
```

## Troubleshooting

### Error: missing label column

This means you used `--task classify` but your CSV does not include the label column you passed in `--label-column`.

### Error: missing text columns

This means the CSV headers do not match the names passed into `--text-columns`.

### Packages not found

Re-activate the virtual environment and install dependencies again:

```bash
source .venv/bin/activate
pip install -r requirements_topic_model.txt
```

## Recommendation

For your use case with around 5,000 rows:

- if you already know the target topics, use supervised classification
- if you do not have labels yet, start with topic discovery, inspect the discovered groups, then manually assign topic names and retrain as supervised classification
