from transformers import pipeline

# Load the multilingual classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli") 
# Pro-tip: For better results in Cebuano/Tagalog, use 'vicgalle/xlm-roberta-large-xnli'

text = "kailangan ko ayusin ang paglalaro ng chess para makapasa ako sa school"
candidate_labels = ["lifestyle", "food", "sports", "news", "laws"]

result = classifier(text, candidate_labels=candidate_labels)

print(f"Top Topic: {result['labels'][0]} ({result['scores'][0]*100:.2f}%)")